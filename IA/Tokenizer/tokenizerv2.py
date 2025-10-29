from tqdm import tqdm
import argparse
import pickle
import os
import time
from collections import Counter
from datasets import load_dataset
import numpy as np

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("⚠️  Numba non disponible. Installez avec: pip install numba")
    print("   Le script fonctionnera mais sera plus lent.\n")
    # Dummy decorator si numba n'est pas installé
    def jit(*args, **kwargs):
        def wrapper(func):
            return func
        return wrapper
    prange = range


@jit(nopython=True, parallel=True)
def count_pairs_fast(tokens):
    """Compte les paires avec Numba (parallélisé)."""
    n = len(tokens)
    # Utiliser un dict simple pour Numba
    max_val = max(tokens) + 1
    pair_matrix = np.zeros((max_val, max_val), dtype=np.int64)
    
    for i in prange(n - 1):
        pair_matrix[tokens[i], tokens[i+1]] += 1
    
    return pair_matrix


@jit(nopython=True)
def merge_pair_fast(tokens, a, b, new_id):
    """Merge ultra-rapide avec Numba."""
    result = []
    i = 0
    n = len(tokens)
    
    while i < n:
        if i < n - 1 and tokens[i] == a and tokens[i+1] == b:
            result.append(new_id)
            i += 2
        else:
            result.append(tokens[i])
            i += 1
    
    return np.array(result, dtype=np.int32)


class MYBPE():
    def __init__(self, vocab_size, dataset=None):
        self.vocab_size = vocab_size
        if dataset is not None:
            self.dataset = np.array(list(dataset.encode("utf-8")), dtype=np.int32)
    
    def train_tokenizer_numba(self, checkpoint_path=None, checkpoint_freq=500, verbose=True):
        """
        🚀 VERSION NUMBA: Compilation JIT pour vitesse maximale
        Devrait être 10-100x plus rapide que la version Python pure
        """
        if not NUMBA_AVAILABLE:
            print("⚠️  Numba non disponible, utilisation de la version lente...")
            return self.train_tokenizer_fallback(checkpoint_path, checkpoint_freq, verbose)
        
        num_merged_tokens = self.vocab_size - 256
        tokens = self.dataset.copy()
        self.merging_rules = {}
        
        if verbose:
            print("📊 Initialisation Numba (compilation JIT)...")
        
        start_time = time.time()
        last_checkpoint_time = start_time
        compilation_done = False
        
        if verbose:
            with tqdm(total=num_merged_tokens, 
                      desc="🚀 BPE NUMBA (CPU optimisé)",
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                      colour='green') as pbar:
                
                for i in range(num_merged_tokens):
                    # Compter les paires (compilé par Numba)
                    if i % 50 == 0 or i == 0:
                        pair_matrix = count_pairs_fast(tokens)
                        
                        # Trouver le max
                        max_count = 0
                        best_pair = (0, 0)
                        
                        for a in range(pair_matrix.shape[0]):
                            for b in range(pair_matrix.shape[1]):
                                if pair_matrix[a, b] > max_count:
                                    max_count = pair_matrix[a, b]
                                    best_pair = (a, b)
                        
                        if max_count == 0:
                            pbar.write(f"⚠️  Plus de paires (itération {i})")
                            break
                        
                        top_pair = best_pair
                        count = max_count
                    
                    new_token_id = i + 256
                    
                    # Afficher les stats
                    if i == 10 and not compilation_done:
                        compilation_time = time.time() - start_time
                        pbar.write(f"✅ Compilation JIT terminée ({compilation_time:.1f}s)")
                        compilation_done = True
                    
                    if i % 20 == 0 and i > 0:
                        elapsed = time.time() - start_time
                        avg_time = elapsed / i
                        eta_seconds = avg_time * (num_merged_tokens - i)
                        eta_str = f"{int(eta_seconds//3600)}h{int((eta_seconds%3600)//60)}m"
                        
                        pbar.set_postfix({
                            'freq': f"{count:,}",
                            'tokens': f"{len(tokens):,}",
                            'ETA': eta_str,
                            'it/s': f"{i/elapsed:.2f}"
                        })
                    
                    # Merger (compilé par Numba)
                    tokens = merge_pair_fast(tokens, top_pair[0], top_pair[1], new_token_id)
                    self.merging_rules[top_pair] = new_token_id
                    
                    pbar.update(1)
                    
                    # Checkpoints
                    if checkpoint_path and (i + 1) % checkpoint_freq == 0:
                        checkpoint_time = time.time()
                        time_since_last = checkpoint_time - last_checkpoint_time
                        temp_path = f"{checkpoint_path}.checkpoint_{i+1}"
                        
                        self.build_vocabulary(silent=True)
                        self.save_tokenizer(temp_path, silent=True)
                        
                        pbar.write(f"💾 Checkpoint {i+1} ({time_since_last:.1f}s)")
                        last_checkpoint_time = time.time()
            
            total_time = time.time() - start_time
            print(f"\n✨ Terminé en {int(total_time//3600)}h{int((total_time%3600)//60)}m{int(total_time%60)}s")
            print(f"📝 {len(self.merging_rules):,} règles")
            print(f"⚡ Vitesse: {num_merged_tokens/(total_time/3600):.0f} merges/heure")
        
        return self.merging_rules
    
    def train_tokenizer_fallback(self, checkpoint_path=None, checkpoint_freq=500, verbose=True):
        """Version de secours sans Numba."""
        num_merged_tokens = self.vocab_size - 256
        tokens = list(self.dataset)
        self.merging_rules = {}
        
        start_time = time.time()
        
        if verbose:
            with tqdm(total=num_merged_tokens, desc="🚀 BPE (sans Numba)", colour='yellow') as pbar:
                for i in range(num_merged_tokens):
                    if i % 50 == 0:
                        pairs = Counter()
                        for j in range(len(tokens) - 1):
                            pairs[(tokens[j], tokens[j+1])] += 1
                        
                        if not pairs:
                            break
                        
                        top_pair = max(pairs, key=pairs.get)
                    
                    new_id = i + 256
                    
                    # Merge
                    result = []
                    j = 0
                    while j < len(tokens):
                        if j < len(tokens) - 1 and tokens[j] == top_pair[0] and tokens[j+1] == top_pair[1]:
                            result.append(new_id)
                            j += 2
                        else:
                            result.append(tokens[j])
                            j += 1
                    
                    tokens = result
                    self.merging_rules[top_pair] = new_id
                    pbar.update(1)
        
        return self.merging_rules
    
    def train_tokenizer(self, checkpoint_path=None, checkpoint_freq=500, verbose=True):
        """Wrapper"""
        return self.train_tokenizer_numba(checkpoint_path, checkpoint_freq, verbose)
    
    def build_vocabulary(self, silent=False):
        """Build vocabulary."""
        self.voc = {i: bytes([i]) for i in range(256)}
        
        if silent:
            for pair, val in self.merging_rules.items():
                self.voc[val] = self.voc[pair[0]] + self.voc[pair[1]]
        else:
            for pair, val in tqdm(self.merging_rules.items(), desc="📚 Vocab", colour='blue'):
                self.voc[val] = self.voc[pair[0]] + self.voc[pair[1]]
            print(f"✅ {len(self.voc):,} tokens")
    
    def save_tokenizer(self, path, silent=False):
        """Save."""
        with open(path, "wb") as f:
            pickle.dump({
                "merging_rules": self.merging_rules,
                "vocabulary": self.voc,
                "vocab_size": self.vocab_size
            }, f)
        if not silent:
            print(f"💾 {path}")
    
    def load_tokenizer(self, path, verbose=True):
        """Load."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.merging_rules = data["merging_rules"]
        self.voc = data["vocabulary"]
        if "vocab_size" in data:
            self.vocab_size = data["vocab_size"]
        if verbose:
            print(f"✅ {len(self.voc):,} tokens")
    
    def decoder(self, ids):
        """Decode."""
        text = b"".join(self.voc[i] for i in ids)
        return text.decode("utf-8", errors="replace")
    
    def encoder(self, text):
        """Encode."""
        byte_tokens = list(text.encode("utf-8"))
        merge_priority = {pair: idx for idx, pair in enumerate(self.merging_rules.keys())}
        
        while len(byte_tokens) > 1:
            pairs = Counter()
            for i in range(len(byte_tokens) - 1):
                pairs[(byte_tokens[i], byte_tokens[i+1])] += 1
            
            replace_pair = min(pairs.keys(), key=lambda p: merge_priority.get(p, float('inf')), default=None)
            
            if replace_pair is None or replace_pair not in self.merging_rules:
                break
            
            result = []
            i = 0
            while i < len(byte_tokens):
                if i < len(byte_tokens) - 1 and byte_tokens[i] == replace_pair[0] and byte_tokens[i+1] == replace_pair[1]:
                    result.append(self.merging_rules[replace_pair])
                    i += 2
                else:
                    result.append(byte_tokens[i])
                    i += 1
            byte_tokens = result
        
        return byte_tokens


def load_c4_dataset(language="en", max_samples=50000):
    """Charge C4."""
    print(f"\n📥 C4-{language} ({max_samples:,} samples)")
    ds = load_dataset("allenai/c4", language, split="train", streaming=True)
    
    texts = []
    for i, ex in enumerate(tqdm(ds, total=max_samples, desc=f"C4-{language}")):
        if i >= max_samples:
            break
        if 'text' in ex:
            texts.append(ex['text'])
    
    full_text = "\n\n".join(texts)
    print(f"✅ {len(full_text):,} chars ({len(full_text)/1e6:.1f} MB)")
    return full_text


def load_mixed_c4_dataset(en_samples=30000, fr_samples=20000):
    """C4 EN+FR."""
    print("\n" + "="*70)
    print("📚 C4 MULTILINGUE")
    print("="*70)
    
    en_text = load_c4_dataset("en", en_samples)
    fr_text = load_c4_dataset("fr", fr_samples)
    
    combined = en_text + "\n\n" + fr_text
    print(f"\n✅ Total: {len(combined):,} chars ({len(combined)/1e6:.1f} MB)")
    return combined


def load_single_hf_dataset(dataset_name, split="train", max_samples=None):
    """Dataset HF."""
    print(f"\n📥 {dataset_name}")
    ds = load_dataset(dataset_name, split=split)
    
    texts = []
    samples = min(len(ds), max_samples) if max_samples else len(ds)
    
    for i, ex in enumerate(tqdm(ds, total=samples, desc="Extraction")):
        if max_samples and i >= max_samples:
            break
        
        if 'messages' in ex:
            for msg in ex['messages']:
                if 'content' in msg:
                    texts.append(msg['content'])
        elif 'text' in ex:
            texts.append(ex['text'])
        elif 'prompt' in ex:
            texts.append(ex['prompt'])
            if 'completion' in ex:
                texts.append(ex['completion'])
    
    full_text = "\n\n".join(texts)
    print(f"✅ {len(full_text):,} chars ({len(full_text)/1e6:.1f} MB)")
    return full_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="🚀 Tokenizer BPE NUMBA")
    
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--use_c4", action="store_true")
    parser.add_argument("--en_samples", type=int, default=20000)
    parser.add_argument("--fr_samples", type=int, default=5000)
    parser.add_argument("--hf_dataset", type=str, default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--hf_split", type=str, default="train_sft")
    parser.add_argument("--max_samples", type=int, default=10000)
    parser.add_argument("--save", default="./tokenizer_50k.bin")
    parser.add_argument("--load", default="./tokenizer_50k.bin")
    parser.add_argument("--use_tokenizer", action="store_true")
    parser.add_argument("--vocab_size", default=50000, type=int)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--checkpoint_freq", type=int, default=1000)
    parser.add_argument("--input", type=str)
    
    args = parser.parse_args()
    
    if args.train:
        print("\n" + "="*70)
        print("🚀 TOKENIZER NUMBA (CPU OPTIMISÉ)")
        print("="*70)
        
        if not NUMBA_AVAILABLE:
            print("\n⚠️  Installez Numba pour 10-100x plus de vitesse:")
            print("   pip install numba\n")
        
        data = None
        
        if args.dataset and os.path.exists(args.dataset):
            with open(args.dataset, "r", encoding="utf-8") as f:
                data = f.read()
        elif args.use_c4:
            data = load_mixed_c4_dataset(args.en_samples, args.fr_samples)
        else:
            data = load_single_hf_dataset(args.hf_dataset, args.hf_split, args.max_samples)
        
        if not data:
            print("❌ Pas de données!")
            exit(1)
        
        print(f"\n📊 {len(data):,} chars ({len(data)/1e6:.1f} MB)")
        print(f"🎯 Vocab: {args.vocab_size:,} | Merges: {args.vocab_size-256:,}\n")
        
        tokenizer = MYBPE(args.vocab_size, data)
        tokenizer.train_tokenizer(args.checkpoint, args.checkpoint_freq, verbose=True)
        tokenizer.build_vocabulary()
        tokenizer.save_tokenizer(args.save)
        
        print(f"\n🎉 Sauvegardé: {args.save}")
    
    elif args.use_tokenizer:
        tokenizer = MYBPE(args.vocab_size)
        tokenizer.load_tokenizer(args.load, verbose=True)
        
        if os.path.isfile(args.input):
            with open(args.input, "r", encoding="utf-8") as f:
                input_data = f.read()
        else:
            input_data = args.input
        
        tokens = tokenizer.encoder(input_data)
        print(f"🎯 {len(tokens)} tokens")
    
    else:
        print("❌ Utilisez --train ou --use_tokenizer")
        exit(1)

#python tokenizerv2.py --train --vocab_size 50000 --max_samples 10000 --save tokenizer_50k.bin