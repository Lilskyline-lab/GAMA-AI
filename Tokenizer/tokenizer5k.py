from tqdm import tqdm
import argparse
import pickle
import os
import time
from collections import Counter

class MYBPE():
    def __init__(self, vocab_size, dataset=None):
        self.vocab_size = vocab_size
        if dataset is not None:
            self.dataset = list(dataset.encode("utf-8"))
    
    def get_pairs(self, dataset):
        """Generate and count adjacent token pairs efficiently using Counter."""
        pairs = Counter()
        for i in range(len(dataset) - 1):
            pairs[(dataset[i], dataset[i+1])] += 1
        return pairs
    
    def merge_tokens(self, tokens, pair, new_id):
        """Replace all occurrences of a token pair with a new token id."""
        merged_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
                merged_tokens.append(new_id)
                i += 2
            else:
                merged_tokens.append(tokens[i])
                i += 1
        return merged_tokens
    
    def train_tokenizer(self, checkpoint_path=None, checkpoint_freq=500, verbose=True):
        """Train the BPE tokenizer by iteratively merging frequent token pairs."""
        num_merged_tokens = self.vocab_size - 256
        tokens = self.dataset
        self.merging_rules = {}
        
        start_time = time.time()
        last_checkpoint_time = start_time
        
        if verbose:
            # Mode verbeux : barre détaillée
            with tqdm(total=num_merged_tokens, 
                      desc="🔥 Entraînement BPE",
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                      colour='green') as pbar:
                
                for i in range(num_merged_tokens):
                    pair_details = self.get_pairs(tokens)
                    
                    if not pair_details:
                        pbar.write(f"⚠️  Plus de paires à fusionner (itération {i})")
                        break
                    
                    top_pair = max(pair_details, key=pair_details.get)
                    new_token_id = i + 256
                    
                    if i > 0:
                        elapsed = time.time() - start_time
                        avg_time_per_iter = elapsed / i
                        eta_seconds = avg_time_per_iter * (num_merged_tokens - i)
                        eta_str = f"{int(eta_seconds//3600)}h{int((eta_seconds%3600)//60)}m"
                    else:
                        eta_str = "calcul..."
                    
                    pbar.set_postfix({
                        '🔢 freq': f"{pair_details[top_pair]:,}",
                        '📊 tokens': f"{len(tokens):,}",
                        '⏱️ reste': eta_str
                    })
                    
                    tokens = self.merge_tokens(tokens, top_pair, new_token_id)
                    self.merging_rules[top_pair] = new_token_id
                    pbar.update(1)
                    
                    if checkpoint_path and (i + 1) % checkpoint_freq == 0:
                        checkpoint_time = time.time()
                        time_since_last = checkpoint_time - last_checkpoint_time
                        temp_path = f"{checkpoint_path}.checkpoint_{i+1}"
                        
                        self.build_vocabulary(silent=True)
                        self.save_tokenizer(temp_path, silent=True)
                        
                        pbar.write(f"💾 Checkpoint {i+1} sauvegardé ({time_since_last:.1f}s)")
                        last_checkpoint_time = time.time()
            
            total_time = time.time() - start_time
            print(f"\n✨ Entraînement terminé en {int(total_time//3600)}h{int((total_time%3600)//60)}m{int(total_time%60)}s")
            print(f"📝 {len(self.merging_rules):,} règles de fusion créées")
        else:
            # Mode silencieux : juste une barre simple
            for i in range(num_merged_tokens):
                pair_details = self.get_pairs(tokens)
                if not pair_details:
                    break
                
                top_pair = max(pair_details, key=pair_details.get)
                new_token_id = i + 256
                tokens = self.merge_tokens(tokens, top_pair, new_token_id)
                self.merging_rules[top_pair] = new_token_id
                
                if checkpoint_path and (i + 1) % checkpoint_freq == 0:
                    temp_path = f"{checkpoint_path}.checkpoint_{i+1}"
                    self.build_vocabulary(silent=True)
                    self.save_tokenizer(temp_path, silent=True)
        
        return self.merging_rules
    
    def build_vocabulary(self, silent=False):
        """Build the vocabulary mapping from token ID to actual byte sequence."""
        self.voc = {i: bytes([i]) for i in range(256)}
        
        if silent:
            for pair, val in self.merging_rules.items():
                self.voc[val] = self.voc[pair[0]] + self.voc[pair[1]]
        else:
            with tqdm(self.merging_rules.items(), 
                      desc="📚 Construction vocabulaire",
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}',
                      colour='blue') as pbar:
                for pair, val in pbar:
                    self.voc[val] = self.voc[pair[0]] + self.voc[pair[1]]
            
            print(f"✅ Vocabulaire: {len(self.voc):,} tokens")
    
    def save_tokenizer(self, path, silent=False):
        """Save tokenizer with additional metadata."""
        with open(path, "wb") as f:
            pickle.dump({
                "merging_rules": self.merging_rules,
                "vocabulary": self.voc,
                "vocab_size": self.vocab_size
            }, f)
        if not silent:
            print(f"💾 Tokenizer sauvegardé: {path}")
    
    def load_tokenizer(self, path, verbose=True):
        """Load tokenizer with validation."""
        if verbose:
            with tqdm(total=1, desc="📂 Chargement tokenizer", colour='magenta', leave=False) as pbar:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                self.merging_rules = data["merging_rules"]
                self.voc = data["vocabulary"]
                if "vocab_size" in data:
                    self.vocab_size = data["vocab_size"]
                pbar.update(1)
            print(f"✅ Tokenizer chargé | Vocab: {len(self.voc):,} tokens")
        else:
            # Mode silencieux : pas de barre, pas de print
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.merging_rules = data["merging_rules"]
            self.voc = data["vocabulary"]
            if "vocab_size" in data:
                self.vocab_size = data["vocab_size"]
    
    def decoder(self, ids, verbose=False):
        """Decode a list of token IDs into a UTF-8 string using the vocabulary."""
        text = b"".join(self.voc[i] for i in ids)
        text = text.decode("utf-8", errors="replace")
        return text
    
    def encoder(self, text, verbose=False):
        """Encode raw UTF-8 text into token IDs using trained merges."""
        byte_tokens = list(text.encode("utf-8"))
        merge_priority = {pair: idx for pair, idx in self.merging_rules.items()}
        
        if verbose:
            # Mode verbeux : affiche la progression
            merge_count = 0
            with tqdm(desc="🔤 Encodage", 
                      bar_format='{desc}: {n_fmt} fusions | {elapsed}',
                      colour='yellow') as pbar:
                while len(byte_tokens) > 1:
                    pairs = self.get_pairs(byte_tokens)
                    
                    replace_pair = min(
                        pairs.keys(), 
                        key=lambda p: merge_priority.get(p, float('inf')),
                        default=None
                    )
                    
                    if replace_pair is None or replace_pair not in self.merging_rules:
                        break
                    
                    byte_tokens = self.merge_tokens(
                        byte_tokens, 
                        replace_pair, 
                        self.merging_rules[replace_pair]
                    )
                    merge_count += 1
                    pbar.update(1)
            
            print(f"✅ Encodage terminé: {len(byte_tokens)} tokens")
        else:
            # Mode silencieux : pas d'affichage
            while len(byte_tokens) > 1:
                pairs = self.get_pairs(byte_tokens)
                
                replace_pair = min(
                    pairs.keys(), 
                    key=lambda p: merge_priority.get(p, float('inf')),
                    default=None
                )
                
                if replace_pair is None or replace_pair not in self.merging_rules:
                    break
                
                byte_tokens = self.merge_tokens(
                    byte_tokens, 
                    replace_pair, 
                    self.merging_rules[replace_pair]
                )
        
        return byte_tokens

def valid_tokenizer_model(name: str):
    """Validate tokenizer model file."""
    if not name.endswith(".bin"):
        raise argparse.ArgumentTypeError("Le fichier doit avoir l'extension '.bin'")
    
    if os.path.exists(name):
        try:
            with open(name, "rb") as f:
                data = pickle.load(f)
            required_keys = ["merging_rules", "vocabulary"]
            if not isinstance(data, dict) or not all(k in data for k in required_keys):
                raise argparse.ArgumentTypeError(
                    "Le fichier .bin doit contenir 'merging_rules' et 'vocabulary'"
                )
        except Exception as e:
            raise argparse.ArgumentTypeError(f"Fichier .bin invalide: {e}")
    return name

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="🚀 Tokenizer BPE Optimisé",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  # Entraîner un tokenizer (5000 tokens vocab)
  python tokenizer.py --train --vocab_size 5000 --dataset train.txt --save tokenizer_5k.bin
  
  # Avec checkpoints (recommandé pour gros datasets)
  python tokenizer.py --train --vocab_size 5000 --dataset data.txt --save tok.bin --checkpoint cp
  
  # Encoder du texte
  python tokenizer.py --use_tokenizer --load tokenizer_5k.bin --input "Bonjour le monde!"
  
  # Encoder un fichier
  python tokenizer.py --use_tokenizer --load tokenizer_5k.bin --input input.txt
        """
    )
    
    parser.add_argument("--dataset", type=str, default="./train.txt",
                        help="Dataset d'entraînement (fichier texte)")
    parser.add_argument("--save", default="./tokenizer_model.bin", type=valid_tokenizer_model,
                        help="Chemin de sauvegarde du tokenizer")
    parser.add_argument("--load", default="./tokenizer_model.bin", type=valid_tokenizer_model,
                        help="Chemin de chargement du tokenizer")
    parser.add_argument("--use_tokenizer", action="store_true",
                        help="Utiliser le tokenizer sur un input")
    parser.add_argument("--vocab_size", default=5000, type=int,
                        help="Taille du vocabulaire (>= 256, défaut: 5000)")
    parser.add_argument("--train", action="store_true",
                        help="Entraîner un nouveau tokenizer")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Activer les checkpoints - sauvegarde périodique")
    parser.add_argument("--checkpoint_freq", type=int, default=500,
                        help="Fréquence des checkpoints (défaut: 500 itérations)")
    parser.add_argument("--input", type=str,
                        help="Chemin fichier ou texte brut à tokeniser")
    
    args = parser.parse_args()
    
    if args.vocab_size < 256:
        parser.error("vocab_size doit être >= 256")
    
    if args.train:
        print("\n" + "="*60)
        print("🎯 MODE ENTRAÎNEMENT")
        print("="*60)
        
        # Chargement du dataset avec barre de progression
        with tqdm(total=1, desc="📖 Chargement dataset", colour='cyan') as pbar:
            with open(args.dataset, "r", encoding="utf-8") as f:
                data = f.read()
            pbar.update(1)
        
        data_size_mb = len(data) / 1e6
        print(f"📊 Dataset: {len(data):,} caractères ({data_size_mb:.2f} MB)")
        print(f"🎯 Vocab cible: {args.vocab_size:,} tokens")
        print(f"🔄 Fusions nécessaires: {args.vocab_size - 256:,}\n")
        
        tokenizer = MYBPE(args.vocab_size, data)
        tokenizer.train_tokenizer(checkpoint_path=args.checkpoint, checkpoint_freq=args.checkpoint_freq, verbose=True)
        tokenizer.build_vocabulary()
        tokenizer.save_tokenizer(args.save)
        
        print("\n" + "="*60)
        print("🎉 ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
        print("="*60 + "\n")
    
    if args.use_tokenizer:
        print("\n" + "="*60)
        print("🔤 MODE UTILISATION")
        print("="*60 + "\n")
        
        tokenizer = MYBPE(args.vocab_size)
        tokenizer.load_tokenizer(args.load, verbose=True)
        
        if not args.input:
            parser.error("--input requis avec --use_tokenizer")
        
        if os.path.isfile(args.input):
            with tqdm(total=1, desc="📖 Lecture fichier", colour='cyan') as pbar:
                with open(args.input, "r", encoding="utf-8") as f:
                    input_data = f.read()
                pbar.update(1)
            print(f"📄 Fichier chargé: {len(input_data):,} caractères\n")
        else:
            input_data = args.input
            print(f"📝 Texte direct: {len(input_data)} caractères\n")
        
        tokens = tokenizer.encoder(input_data, verbose=True)
        print(f"\n🎯 Résultat: {len(tokens)} tokens générés")