"""
Script de test pour le modèle entraîné avec LoRA
Permet de tester le modèle en mode interactif ou avec des prompts prédéfinis
"""

import os
import sys
import json
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model.gpt2_model import GPT2Model
from Tokenizer.tokenizer5k import MYBPE


class LoRALayer(nn.Module):
    """Couche LoRA (même implémentation que dans l'entraînement)"""
    def __init__(self, in_features, out_features, rank=8, alpha=16, dropout=0.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x, original_output):
        lora_output = self.dropout(x) @ self.lora_A @ self.lora_B
        return original_output + lora_output * self.scaling


class LoRAWrapper(nn.Module):
    """Wrapper LoRA pour l'inférence"""
    def __init__(self, base_model, rank=8, alpha=16, dropout=0.0, target_modules=None):
        super().__init__()
        self.base_model = base_model
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        
        if target_modules is None:
            target_modules = ['q_proj', 'k_proj', 'v_proj', 'fc1', 'fc2']
        self.target_modules = target_modules
        
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        self.lora_layers = nn.ModuleDict()
        self.module_references = {}
        self._inject_lora()
    
    def _inject_lora(self):
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Linear):
                for target in self.target_modules:
                    if target in name:
                        lora_layer = LoRALayer(
                            module.in_features,
                            module.out_features,
                            rank=self.rank,
                            alpha=self.alpha,
                            dropout=self.dropout
                        )
                        safe_name = name.replace('.', '_')
                        self.lora_layers[safe_name] = lora_layer
                        self.module_references[safe_name] = module
                        break
    
    def forward(self, input_ids, attention_mask=None):
        handles = []
        
        def make_hook(lora_layer):
            def hook(module, input, output):
                return lora_layer(input[0], output)
            return hook
        
        for safe_name, lora_layer in self.lora_layers.items():
            module = self.module_references[safe_name]
            handle = module.register_forward_hook(make_hook(lora_layer))
            handles.append(handle)
        
        logits, hidden_states = self.base_model(input_ids)
        
        for handle in handles:
            handle.remove()
        
        return logits, hidden_states
    
    def load_lora_weights(self, path):
        lora_state = torch.load(path, map_location=next(self.parameters()).device)
        self.lora_layers.load_state_dict(lora_state['lora_layers'])
        print(f"✅ Poids LoRA chargés: {path}")


class LoRAInference:
    """Classe pour faire de l'inférence avec le modèle LoRA"""
    
    def __init__(self, model_dir, tokenizer_path, device='cpu'):
        self.model_dir = model_dir
        self.device = torch.device(device)
        
        print("\n" + "="*70)
        print("🤖 CHARGEMENT DU MODÈLE LoRA")
        print("="*70)
        
        # Charger la configuration
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        print(f"\n📋 Configuration:")
        print(f"   - Vocab size: {self.config['vocab_size']}")
        print(f"   - Embed dim: {self.config['embed_dim']}")
        print(f"   - Num layers: {self.config['num_layers']}")
        print(f"   - Max seq len: {self.config['max_seq_len']}")
        
        # Charger le tokenizer
        self.tokenizer = MYBPE(vocab_size=self.config['vocab_size'])
        self.tokenizer.load_tokenizer(tokenizer_path)
        print(f"\n✅ Tokenizer chargé: {self.config['vocab_size']} tokens")
        
        # Charger le modèle de base
        base_model = GPT2Model(
            vocab_size=self.config['vocab_size'],
            embed_dim=self.config['embed_dim'],
            num_heads=self.config['num_heads'],
            num_layers=self.config['num_layers'],
            max_seq_len=self.config['max_seq_len']
        )
        
        model_path = os.path.join(model_dir, "model.pt")
        if os.path.exists(model_path):
            state = torch.load(model_path, map_location=self.device, weights_only=False)
            base_model.load_state_dict(state)
            print(f"✅ Modèle de base chargé")
        
        base_model.to(self.device)
        
        # Charger la configuration LoRA
        lora_config_path = os.path.join(model_dir, "lora_config.json")
        if os.path.exists(lora_config_path):
            with open(lora_config_path, 'r') as f:
                lora_config = json.load(f)
        else:
            lora_config = {"rank": 8, "alpha": 16, "dropout": 0.0}
        
        print(f"\n🔧 Configuration LoRA:")
        print(f"   - Rank: {lora_config['rank']}")
        print(f"   - Alpha: {lora_config['alpha']}")
        
        # Créer le wrapper LoRA
        self.model = LoRAWrapper(
            base_model,
            rank=lora_config['rank'],
            alpha=lora_config['alpha'],
            dropout=0.0,  # Pas de dropout en inférence
            target_modules=lora_config.get('target_modules')
        )
        
        # Charger les poids LoRA
        lora_weights_path = os.path.join(model_dir, "lora_weights.pt")
        if os.path.exists(lora_weights_path):
            self.model.load_lora_weights(lora_weights_path)
        
        self.model.eval()
        
        print("\n✅ Modèle prêt pour l'inférence!")
        print("="*70)
    
    def generate(
        self, 
        prompt, 
        max_length=100, 
        temperature=0.8, 
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2
    ):
        """
        Génère du texte à partir d'un prompt
        
        Args:
            prompt: Texte d'entrée
            max_length: Longueur maximale de génération
            temperature: Contrôle la créativité (0.1=conservateur, 1.5=créatif)
            top_k: Nombre de tokens à considérer
            top_p: Nucleus sampling
            repetition_penalty: Pénalité pour les répétitions
        """
        # Format instruction tuning
        formatted_prompt = f"Human: {prompt}\nBot:"
        
        # Tokeniser
        input_ids = self.tokenizer.encoder(formatted_prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        
        generated = input_ids.copy()
        token_counts = {}  # Pour la pénalité de répétition
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                logits, _ = self.model(input_tensor)
                next_token_logits = logits[0, -1, :] / temperature
                
                # Appliquer la pénalité de répétition
                for token_id, count in token_counts.items():
                    next_token_logits[token_id] /= (repetition_penalty ** count)
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                
                # Mise à jour du compteur de répétition
                token_counts[next_token] = token_counts.get(next_token, 0) + 1
                
                # Ajouter le token généré
                generated.append(next_token)
                
                # Mettre à jour l'input
                input_tensor = torch.tensor([generated], dtype=torch.long).to(self.device)
                
                # Tronquer si trop long
                if len(generated) > self.config['max_seq_len']:
                    generated = generated[-self.config['max_seq_len']:]
                    input_tensor = input_tensor[:, -self.config['max_seq_len']:]
                
                # Stop si on génère un token spécial (à adapter selon votre tokenizer)
                if next_token == 0:  # Supposons que 0 est le token de fin
                    break
        
        # Décoder
        generated_text = self.tokenizer.decoder(generated)
        
        # Extraire seulement la réponse du bot
        if "Bot:" in generated_text:
            response = generated_text.split("Bot:")[-1].strip()
        else:
            response = generated_text[len(formatted_prompt):].strip()
        
        return response
    
    def interactive_mode(self):
        """Mode interactif pour tester le modèle"""
        print("\n" + "="*70)
        print("💬 MODE INTERACTIF")
        print("="*70)
        print("\nCommandes spéciales:")
        print("  - 'quit' ou 'exit' : Quitter")
        print("  - 'clear' : Effacer l'écran")
        print("  - 'temp <valeur>' : Changer la température (ex: temp 1.0)")
        print("  - 'length <valeur>' : Changer la longueur max (ex: length 150)")
        print("\n" + "-"*70)
        
        temperature = 0.8
        max_length = 100
        
        while True:
            try:
                prompt = input("\n👤 Vous: ").strip()
                
                if not prompt:
                    continue
                
                if prompt.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Au revoir!")
                    break
                
                if prompt.lower() == 'clear':
                    os.system('clear' if os.name != 'nt' else 'cls')
                    continue
                
                if prompt.lower().startswith('temp '):
                    try:
                        temperature = float(prompt.split()[1])
                        print(f"✅ Température définie à {temperature}")
                        continue
                    except:
                        print("❌ Format invalide. Utilisez: temp 0.8")
                        continue
                
                if prompt.lower().startswith('length '):
                    try:
                        max_length = int(prompt.split()[1])
                        print(f"✅ Longueur max définie à {max_length}")
                        continue
                    except:
                        print("❌ Format invalide. Utilisez: length 100")
                        continue
                
                # Générer la réponse
                print(f"\n🤖 Bot: ", end='', flush=True)
                response = self.generate(
                    prompt,
                    max_length=max_length,
                    temperature=temperature
                )
                print(response)
                
            except KeyboardInterrupt:
                print("\n\n👋 Au revoir!")
                break
            except Exception as e:
                print(f"\n❌ Erreur: {e}")
    
    def test_prompts(self, test_cases):
        """Teste le modèle avec une liste de prompts prédéfinis"""
        print("\n" + "="*70)
        print("🧪 TEST DU MODÈLE")
        print("="*70)
        
        for i, prompt in enumerate(test_cases, 1):
            print(f"\n📝 Test {i}/{len(test_cases)}")
            print(f"👤 Prompt: {prompt}")
            print(f"🤖 Réponse: ", end='', flush=True)
            
            response = self.generate(prompt, max_length=80, temperature=0.7)
            print(response)
            print("-" * 70)


def main():
    print("\n" + "="*70)
    print("🤖 SCRIPT DE TEST - MODÈLE LoRA")
    print("="*70)
    
    # Configuration
    model_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "saved_models",
        "my_llm_lora"
    )
    
    tokenizer_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Tokenizer",
        "tokenizer_5k.bin"
    )
    
    if not os.path.exists(tokenizer_path):
        tokenizer_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "Tokenizer",
            "tokenizer_model.bin"
        )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"💻 Device: {device}")
    
    # Charger le modèle
    inference = LoRAInference(model_dir, tokenizer_path, device)
    
    # Menu de choix
    print("\n" + "="*70)
    print("📋 MENU")
    print("="*70)
    print("\n1. Mode interactif (conversation)")
    print("2. Tests prédéfinis")
    print("3. Quitter")
    
    choice = input("\nVotre choix (1-3): ").strip()
    
    if choice == "1":
        inference.interactive_mode()
    
    elif choice == "2":
        test_prompts = [
            "What is artificial intelligence?",
            "Explain quantum computing in simple terms.",
            "Tell me about the history of computers.",
            "How does machine learning work?",
            "What are the benefits of using Python?",
            "Explain what LoRA is.",
            "What is the capital of France?",
            "How can I learn programming?",
        ]
        
        inference.test_prompts(test_prompts)
        
        # Proposer le mode interactif après
        print("\n" + "="*70)
        cont = input("\n💬 Passer en mode interactif ? (o/n): ").strip().lower()
        if cont == 'o':
            inference.interactive_mode()
    
    else:
        print("\n👋 Au revoir!")


if __name__ == "__main__":
    main()