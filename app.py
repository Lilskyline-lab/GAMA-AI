import os
import json
import torch
import io
import sys
import threading
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

sys.path.append('./IA/training/Model')
sys.path.append('./IA/training/TransformerBlock')
sys.path.append('./IA/training/Attention')
sys.path.append('./IA/training/FeedForward')
sys.path.append('./IA/training/Embeddings_Layer')
sys.path.append('./IA/Tokenizer')

from gpt2_model import GPT2Model
from tokenizer5k import MYBPE

app = Flask(__name__)
CORS(app)

model = None
tokenizer = None
config = None
device = None
model_loaded = False
load_lock = threading.Lock()

def silence_output():
    old_out = sys.stdout
    old_err = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    return old_out, old_err

def restore_output(old):
    out, err = old
    sys.stdout = out
    sys.stderr = err

def load_model_and_tokenizer():
    global model, tokenizer, config, device, model_loaded
    
    if model_loaded:
        return
    
    with load_lock:
        if model_loaded:
            return
        
        model_dir = "./IA/saved_models/my_llm"
        tokenizer_path = "./IA/Tokenizer/tokenizer_5k.bin"
        device = torch.device("cpu")
        
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        print("🔤 Chargement du tokenizer...")
        tokenizer = MYBPE(vocab_size=config.get("vocab_size", 5000))
        old = silence_output()
        try:
            tokenizer.load_tokenizer(tokenizer_path, verbose=False)
        finally:
            restore_output(old)
        print("✅ Tokenizer chargé")
        
        print("🤖 Initialisation du modèle...")
        model = GPT2Model(
            vocab_size=config.get("vocab_size", 5000),
            embed_dim=config.get("embed_dim", 256),
            num_heads=config.get("num_heads", 8),
            num_layers=config.get("num_layers", 4),
            max_seq_len=config.get("max_seq_len", 512)
        )
        
        model_file = os.path.join(model_dir, "model.pt")
        if os.path.exists(model_file):
            try:
                state = torch.load(model_file, map_location=device, weights_only=True)
            except TypeError:
                state = torch.load(model_file, map_location=device)
            model.load_state_dict(state)
            print(f"✅ Modèle chargé depuis {model_file}")
        else:
            print(f"⚠️ Fichier model.pt non trouvé. Le modèle utilisera des poids aléatoires.")
        
        model.to(device)
        model.eval()
        model_loaded = True
        print("✅ Modèle prêt!")

def generate_response(prompt, max_new_tokens=40, temperature=0.7, top_k=50, top_p=0.95, repetition_penalty=1.1):
    old = silence_output()
    try:
        tokens = tokenizer.encoder(prompt, verbose=False)
    finally:
        restore_output(old)
    
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    generated_ids = input_ids[0].tolist()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            inp = torch.tensor([generated_ids[-model.max_seq_len:]], device=device)
            logits, _ = model(inp)
            next_logits = logits[0, -1, :].float()
            
            if repetition_penalty != 1.0:
                for t in set(generated_ids):
                    next_logits[t] = next_logits[t] / repetition_penalty
            
            if temperature != 1.0 and temperature > 0:
                next_logits = next_logits / temperature
            
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
    
    old = silence_output()
    try:
        text = tokenizer.decoder(generated_ids, verbose=False)
    finally:
        restore_output(old)
    
    if "Bot:" in text:
        return text.split("Bot:")[-1].strip()
    if prompt in text:
        return text[len(prompt):].strip()
    return text.strip()

@app.route('/')
def index():
    return render_template('index.html')

def ensure_model_loaded():
    if not model_loaded:
        load_model_and_tokenizer()

@app.route('/api/chat', methods=['POST'])
def chat():
    ensure_model_loaded()
    
    try:
        data = request.json
        user_message = data.get('message', '')
        max_tokens = data.get('max_tokens', 40)
        temperature = data.get('temperature', 0.7)
        top_k = data.get('top_k', 50)
        top_p = data.get('top_p', 0.95)
        repetition_penalty = data.get('repetition_penalty', 1.1)
        
        if not user_message:
            return jsonify({'error': 'Message vide'}), 400
        
        prompt = f"Human: {user_message}\nBot:"
        
        response = generate_response(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )
        
        return jsonify({
            'response': response,
            'success': True
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/api/config', methods=['GET'])
def get_config():
    ensure_model_loaded()
    
    return jsonify({
        'vocab_size': config.get('vocab_size'),
        'embed_dim': config.get('embed_dim'),
        'num_heads': config.get('num_heads'),
        'num_layers': config.get('num_layers'),
        'max_seq_len': config.get('max_seq_len')
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Démarrage de l'application...")
    print("="*60 + "\n")
    
    load_model_and_tokenizer()
    
    print("\n" + "="*60)
    print("✅ Serveur prêt!")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
