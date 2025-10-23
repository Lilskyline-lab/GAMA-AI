import os
import json
import torch
import argparse
import io
import sys
import contextlib

# <-- Ajout : s'assurer que le module local Model/Tokenizer est importable
sys.path.append('../Model')
sys.path.append('../Tokenizer')
from gpt2_model import GPT2Model
from Tokenizer import MYBPE

def silence_stdout():
    """Silence stdout et stderr (utile pour tokenizer/tqdm)."""
    old_out = sys.stdout
    old_err = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    return old_out, old_err

def restore_stdout(old):
    out, err = old
    sys.stdout = out
    sys.stderr = err

def load_config(model_dir):
    cfg_path = os.path.join(model_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    # fallback minimal config compatible avec test1.py
    return {
        "vocab_size": 300,
        "embed_dim": 128,
        "num_heads": 4,
        "num_layers": 2,
        "max_seq_len": 512
    }

def load_model_and_tokenizer(model_dir, tokenizer_path, device):
    # charger config
    config = load_config(model_dir)

    # tokenizer
    print("🔤 Chargement du tokenizer...")
    tokenizer = MYBPE(vocab_size=config.get("vocab_size", 300))
    tokenizer.load_tokenizer(tokenizer_path)
    print("✅ Tokenizer chargé")

    # modèle
    print("🤖 Initialisation du modèle...")
    model = GPT2Model(
        vocab_size=config.get("vocab_size", 300),
        embed_dim=config.get("embed_dim", 128),
        num_heads=config.get("num_heads", 4),
        num_layers=config.get("num_layers", 2),
        max_seq_len=config.get("max_seq_len", 64)
    )

    model_file = os.path.join(model_dir, "model.pt")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Fichier de poids introuvable: {model_file}")

    # Essayer de charger en mode "weights_only" pour éviter les warnings/security (PyTorch récent)
    try:
        state = torch.load(model_file, map_location=device, weights_only=True)
    except TypeError:
        # Anciennes versions de torch n'ont pas weights_only -> fallback
        state = torch.load(model_file, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"✅ Modèle chargé depuis {model_file} sur {device}")
    return model, tokenizer, config

def generate_response(model, tokenizer, prompt, device, max_new_tokens=40, temperature=0.8,
                      top_k=50, top_p=0.95, repetition_penalty=1.1, deterministic=False):
    # encoder silencieusement (capture stdout+stderr)
    old = silence_stdout()
    try:
        tokens = tokenizer.encoder(prompt)
    finally:
        restore_stdout(old)

    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

    generated_ids = None

    # Si le modèle propose generate(), tenter plusieurs signatures possibles puis fallback
    if hasattr(model, "generate") and not deterministic:
        # construire kwargs sans None pour éviter TypeError immédiat
        kwargs = {}
        if max_new_tokens is not None:
            kwargs["max_new_tokens"] = max_new_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_k is not None:
            kwargs["top_k"] = top_k
        if top_p is not None:
            kwargs["top_p"] = top_p
        if repetition_penalty is not None:
            kwargs["repetition_penalty"] = repetition_penalty

        try:
            # Essayer la signature la plus complète
            gen = model.generate(input_ids, **kwargs)
            generated_ids = gen[0].tolist()
        except TypeError:
            # Signature non supportée : essayer une version simplifiée (common params)
            try:
                simple_kwargs = {}
                if max_new_tokens is not None:
                    simple_kwargs["max_new_tokens"] = max_new_tokens
                if temperature is not None:
                    simple_kwargs["temperature"] = temperature
                gen = model.generate(input_ids, **simple_kwargs)
                generated_ids = gen[0].tolist()
            except Exception:
                # dernier recours : fallback vers génération autoregressive manuelle
                generated_ids = None
        except Exception:
            # tout autre problème -> fallback
            generated_ids = None

    # Autoregressive manuel si generate() indisponible ou incompatible
    if generated_ids is None:
        generated_ids = input_ids[0].tolist()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                inp = torch.tensor([generated_ids[-model.max_seq_len:]], device=device)
                logits, _ = model(inp)  # logits: (B, L, V)
                next_logits = logits[0, -1, :].float()

                # appliquer repetition penalty
                if repetition_penalty != 1.0:
                    for t in set(generated_ids):
                        next_logits[t] = next_logits[t] / repetition_penalty

                # température
                if temperature != 1.0 and temperature > 0:
                    next_logits = next_logits / temperature

                if deterministic:
                    next_id = int(torch.argmax(next_logits).item())
                else:
                    # top_k sampling
                    if top_k is not None and top_k > 0:
                        values, indices = torch.topk(next_logits, min(top_k, next_logits.size(0)))
                        probs = torch.softmax(values, dim=-1)
                        next_id = int(indices[torch.multinomial(probs, num_samples=1)].item())
                    elif top_p is not None and 0.0 < top_p < 1.0:
                        probs = torch.softmax(next_logits, dim=-1)
                        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                        cumulative = torch.cumsum(sorted_probs, dim=-1)
                        cutoff = (cumulative <= top_p).cpu().numpy()
                        if not cutoff.any():
                            cutoff[0] = True
                        allowed = sorted_indices[cutoff]
                        allowed_probs = probs[allowed]
                        allowed_probs = allowed_probs / allowed_probs.sum()
                        next_id = int(allowed[torch.multinomial(allowed_probs, num_samples=1)].item())
                    else:
                        probs = torch.softmax(next_logits, dim=-1)
                        next_id = int(torch.multinomial(probs, num_samples=1).item())

                generated_ids.append(next_id)

    # décoder silencieusement
    old = silence_stdout()
    try:
        text = tokenizer.decoder(generated_ids)
    finally:
        restore_stdout(old)

    # extraire la partie Bot:
    if "Bot:" in text:
        return text.split("Bot:")[-1].strip()
    if prompt in text:
        return text[len(prompt):].strip()
    return text.strip()

def chat_loop(model, tokenizer, device, max_new_tokens, temperature):
    print("\n" + "="*60)
    print("💬 MODE CHAT — Tapez 'quit' pour quitter")
    print("="*60)
    while True:
        try:
            user = input("Vous: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not user:
            continue
        if user.lower() in ("quit", "exit", "q"):
            break
        prompt = f"Human: {user}\nBot:"
        try:
            resp = generate_response(model, tokenizer, prompt, device, max_new_tokens=max_new_tokens, temperature=temperature)
        except Exception as e:
            resp = f"[Erreur génération] {e}"
        print(f"Bot: {resp}\n")

def main():
    parser = argparse.ArgumentParser(description="Charger un modèle entraîné et chatter")
    parser.add_argument("--model-dir", type=str, default="./my_tiny_chatbot", help="Répertoire contenant model.pt et config.json")
    parser.add_argument("--tokenizer", type=str, default="../Tokenizer/tokenizer_model.bin", help="Chemin vers le tokenizer BPE")
    parser.add_argument("--device", type=str, default="cpu", help="Device (ex: cpu)")
    parser.add_argument("--max-new-tokens", type=int, default=40, help="Nombre de tokens à générer")
    parser.add_argument("--temperature", type=float, default=0.7, help="Température de génération")
    parser.add_argument("--top-k", type=int, default=50, help="top_k sampling (0 pour désactiver)")
    parser.add_argument("--top-p", type=float, default=0.95, help="top_p (nucleus) sampling (0 pour désactiver)")
    parser.add_argument("--repetition-penalty", type=float, default=1.1, help="Penalité de répétition (>1 réduit répétitions)")
    parser.add_argument("--deterministic", action="store_true", help="Utiliser argmax (déterministe) au lieu d'échantillonnage")
    args = parser.parse_args()

    device = torch.device(args.device)

    try:
        model, tokenizer, _ = load_model_and_tokenizer(args.model_dir, args.tokenizer, device)
    except Exception as e:
        print(f"Erreur au chargement: {e}")
        return

    # passer les paramètres au chat loop via closure
    def _generate(prompt):
        return generate_response(
            model, tokenizer, prompt, device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=(args.top_k if args.top_k > 0 else None),
            top_p=(args.top_p if args.top_p > 0 else None),
            repetition_penalty=args.repetition_penalty,
            deterministic=args.deterministic
        )

    # adapter chat_loop pour appeler _generate (ou remplacer fonction existante selon structure)
    chat_loop(model, tokenizer, device, args.max_new_tokens, args.temperature)

if __name__ == "__main__":
    main()