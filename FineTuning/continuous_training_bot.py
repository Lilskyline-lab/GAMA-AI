"""
Système d'entraînement continu : génération automatique de datasets + fine-tuning incrémental
"""

import os
import sys
import json
import time
import requests
import re
from tqdm import tqdm
from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

# Imports locaux
sys.path.append('../Model')
sys.path.append('../Tokenizer')
from gpt2_model import GPT2Model
from Tokenizer import MYBPE


# ============================================
# WIKIPEDIA SCRAPER + Q&A GENERATOR
# ============================================

class WikipediaScraper:
    def __init__(self, language='fr'):
        self.language = language
        self.api_url = f"https://{language}.wikipedia.org/w/api.php"
        self.headers = {"User-Agent": "WikiQABot/1.0"}

    def get_random_articles(self, count=10):
        print(f"\n📥 Récupération de {count} articles Wikipedia...")
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'random',
            'rnnamespace': 0,
            'rnlimit': count
        }
        try:
            response = requests.get(self.api_url, params=params, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            return [{"title": a["title"], "id": a["id"]} for a in data["query"]["random"]]
        except requests.RequestException as e:
            print(f"⚠️ Erreur réseau : {e}")
            return []

    def get_article_content(self, title: str) -> Dict:
        params = {
            'action': 'query',
            'format': 'json',
            'titles': title,
            'prop': 'extracts',
            'explaintext': True,
            'exsectionformat': 'plain'
        }
        try:
            response = requests.get(self.api_url, params=params, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            page = list(data['query']['pages'].values())[0]
            if 'extract' not in page:
                return None
            text = self._clean_text(page['extract'])
            return {'title': title, 'content': text, 'length': len(text)}
        except:
            return None

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'==+ .*? ==+', '', text)
        text = re.sub(r'\n{2,}', '\n', text)
        text = re.sub(r'\s{2,}', ' ', text)
        return text.strip()


class QAGenerator:
    def __init__(self):
        self.wiki_templates = [
            "Qu'est-ce que {subject} ?",
            "Parle-moi de {subject}.",
            "Explique-moi {subject}.",
            "Que sais-tu sur {subject} ?",
            "Décris {subject}.",
        ]
        self.conversation_templates = [
            ("Bonjour", "Bonjour ! Comment vas-tu aujourd'hui ? 😊"),
            ("Salut", "Salut ! Heureux de te revoir !"),
            ("Comment ça va ?", "Je vais super bien, merci ! Et toi ?"),
            ("Merci", "Avec plaisir 😄"),
            ("Bonne nuit", "Bonne nuit 🌙 fais de beaux rêves !"),
        ]

    def _truncate_sentence(self, text: str, max_len=500):
        if len(text) <= max_len:
            return text.strip()
        truncated = text[:max_len]
        end = max(truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?'))
        if end != -1:
            truncated = truncated[:end + 1]
        return truncated.strip()

    def generate_qa_pairs(self, title: str, content: str, max_pairs=3) -> List[Dict]:
        qa_pairs = []
        paragraphs = [p.strip() for p in content.split('\n') if len(p.strip()) > 100]
        for i, paragraph in enumerate(paragraphs[:max_pairs]):
            question = self.wiki_templates[i % len(self.wiki_templates)].format(subject=title)
            answer = self._truncate_sentence(paragraph, 600)
            qa_pairs.append({"human": question, "assistant": answer})
        
        # Ajouter quelques Q&A conversationnelles
        for q, a in self.conversation_templates:
            qa_pairs.append({"human": q, "assistant": a})
        
        return qa_pairs


# ============================================
# DATASET + TRAINING
# ============================================

class ChatDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length=512):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        h = self.pairs[idx]['human'].strip()
        a = self.pairs[idx]['assistant'].strip()
        prefix = f"Human: {h}\nBot:"
        text = prefix + " " + a
        
        ids_prefix = self.tokenizer.encoder(prefix)
        ids_all = self.tokenizer.encoder(text)
        
        if len(ids_all) > self.max_length:
            ids_all = ids_all[-self.max_length:]
        
        assist_start = max(0, len(ids_all) - len(self.tokenizer.encoder(a)))
        return {
            "input_ids": torch.tensor(ids_all, dtype=torch.long),
            "assist_start": assist_start
        }


def collate_fn(batch, pad_id=0):
    input_ids_list = [b["input_ids"] for b in batch]
    assist_starts = [b["assist_start"] for b in batch]
    max_len = max([t.size(0) for t in input_ids_list])
    
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    
    for i, ids in enumerate(input_ids_list):
        L = ids.size(0)
        input_ids[i, :L] = ids
        attention_mask[i, :L] = 1
        start = assist_starts[i]
        labels[i, start:L] = input_ids[i, start:L]
    
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ============================================
# CONTINUOUS TRAINING SYSTEM
# ============================================

class ContinuousTrainer:
    def __init__(self, model_dir, tokenizer_path, device, language='fr'):
        self.model_dir = model_dir
        self.tokenizer_path = tokenizer_path
        self.device = device
        self.scraper = WikipediaScraper(language)
        self.qa_gen = QAGenerator()
        
        # Créer le dossier si nécessaire
        os.makedirs(model_dir, exist_ok=True)
        
        # Charger ou initialiser le modèle
        self.model, self.tokenizer, self.config = self._load_or_init_model()
        
        # Historique de l'entraînement
        self.history_file = os.path.join(model_dir, "training_history.json")
        self.history = self._load_history()

    def _load_or_init_model(self):
        cfg_path = os.path.join(self.model_dir, "config.json")
        model_path = os.path.join(self.model_dir, "model.pt")
        
        # Charger config
        if os.path.exists(cfg_path):
            with open(cfg_path, 'r') as f:
                cfg = json.load(f)
        else:
            cfg = {
                "vocab_size": 300,
                "embed_dim": 128,
                "num_heads": 4,
                "num_layers": 2,
                "max_seq_len": 512
            }
            with open(cfg_path, 'w') as f:
                json.dump(cfg, f, indent=2)
        
        # Charger tokenizer
        tokenizer = MYBPE(vocab_size=cfg["vocab_size"])
        tokenizer.load_tokenizer(self.tokenizer_path)
        
        # Créer modèle
        model = GPT2Model(
            vocab_size=cfg["vocab_size"],
            embed_dim=cfg["embed_dim"],
            num_heads=cfg["num_heads"],
            num_layers=cfg["num_layers"],
            max_seq_len=cfg["max_seq_len"]
        )
        
        # Charger poids si existants
        if os.path.exists(model_path):
            print(f"✅ Chargement du modèle existant : {model_path}")
            try:
                state = torch.load(model_path, map_location=self.device, weights_only=True)
            except TypeError:
                state = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state)
        else:
            print("🆕 Initialisation d'un nouveau modèle")
        
        model.to(self.device)
        return model, tokenizer, cfg

    def _load_history(self):
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return {"cycles": [], "total_qa_trained": 0}

    def _save_history(self):
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)

    def generate_dataset(self, num_articles=10, qa_per_article=3):
        """Génère un nouveau dataset depuis Wikipedia"""
        print("\n" + "="*60)
        print("🔄 GÉNÉRATION NOUVEAU DATASET")
        print("="*60)
        
        articles = self.scraper.get_random_articles(num_articles)
        dataset = []
        
        for article in tqdm(articles, desc="Articles"):
            data = self.scraper.get_article_content(article['title'])
            if not data or data['length'] < 200:
                continue
            qa_pairs = self.qa_gen.generate_qa_pairs(
                data['title'], data['content'], max_pairs=qa_per_article
            )
            dataset.extend(qa_pairs)
            time.sleep(0.3)  # Rate limiting
        
        print(f"✅ Dataset généré : {len(dataset)} paires Q&A")
        return dataset

    def train_on_dataset(self, dataset, epochs=2, batch_size=8, lr=5e-5):
        """Entraîne le modèle sur un dataset"""
        print("\n" + "="*60)
        print("🚀 ENTRAÎNEMENT SUR DATASET")
        print("="*60)
        
        # Split train/val (90/10)
        split = int(len(dataset) * 0.9)
        train_data = dataset[:split]
        val_data = dataset[split:]
        
        train_ds = ChatDataset(train_data, self.tokenizer, max_length=self.config["max_seq_len"])
        val_ds = ChatDataset(val_data, self.tokenizer, max_length=self.config["max_seq_len"])
        
        pad_id = getattr(self.tokenizer, "eos_id", 0)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                                  collate_fn=lambda b: collate_fn(b, pad_id))
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                               collate_fn=lambda b: collate_fn(b, pad_id))
        
        optimizer = AdamW(self.model.parameters(), lr=lr)
        loss_fn = CrossEntropyLoss(ignore_index=-100)
        
        cycle_losses = []
        
        for ep in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {ep}/{epochs}")
            
            for batch in pbar:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                logits, _ = self.model(input_ids)
                lm_logits = logits[:, :-1, :].contiguous()
                lm_labels = labels[:, 1:].contiguous()
                loss = loss_fn(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = total_loss / len(train_loader)
            cycle_losses.append(avg_loss)
            print(f"Epoch {ep} - Train Loss: {avg_loss:.4f}")
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    logits, _ = self.model(input_ids)
                    lm_logits = logits[:, :-1, :].contiguous()
                    lm_labels = labels[:, 1:].contiguous()
                    loss = loss_fn(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
                    val_loss += loss.item()
            
            avg_val = val_loss / len(val_loader) if len(val_loader) else 0.0
            print(f"Epoch {ep} - Val Loss: {avg_val:.4f}")
        
        # Sauvegarder le modèle (écrase l'ancien)
        model_path = os.path.join(self.model_dir, "model.pt")
        torch.save(self.model.state_dict(), model_path)
        print(f"💾 Modèle sauvegardé : {model_path}")
        
        return cycle_losses

    def run_continuous_training(self, num_cycles=5, articles_per_cycle=10, 
                                qa_per_article=3, epochs=2, batch_size=8, lr=5e-5):
        """Exécute la boucle d'entraînement continu"""
        print("\n" + "="*70)
        print("🤖 ENTRAÎNEMENT CONTINU - DÉMARRAGE")
        print("="*70)
        print(f"📊 Cycles prévus : {num_cycles}")
        print(f"📚 Articles par cycle : {articles_per_cycle}")
        print(f"💬 Q&A par article : {qa_per_article}")
        print(f"🔁 Epochs par cycle : {epochs}")
        print("="*70)
        
        for cycle in range(1, num_cycles + 1):
            print(f"\n\n{'='*70}")
            print(f"🔄 CYCLE {cycle}/{num_cycles}")
            print(f"{'='*70}")
            
            # 1. Générer nouveau dataset
            dataset = self.generate_dataset(articles_per_cycle, qa_per_article)
            
            if not dataset:
                print("⚠️ Aucune donnée générée, passage au cycle suivant...")
                continue
            
            # 2. Entraîner sur ce dataset
            losses = self.train_on_dataset(dataset, epochs, batch_size, lr)
            
            # 3. Mettre à jour l'historique
            self.history["cycles"].append({
                "cycle": cycle,
                "num_qa": len(dataset),
                "avg_loss": sum(losses) / len(losses) if losses else 0,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
            self.history["total_qa_trained"] += len(dataset)
            self._save_history()
            
            print(f"\n✅ Cycle {cycle} terminé !")
            print(f"📊 Total Q&A entraînées : {self.history['total_qa_trained']}")
        
        print("\n" + "="*70)
        print("🎉 ENTRAÎNEMENT CONTINU TERMINÉ !")
        print(f"📊 {num_cycles} cycles complétés")
        print(f"💾 Modèle final : {os.path.join(self.model_dir, 'model.pt')}")
        print("="*70)


# ============================================
# MAIN
# ============================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Système d'entraînement continu avec génération automatique")
    parser.add_argument("--model-dir", type=str, default="./my_tiny_chatbot", 
                       help="Dossier du modèle (sera créé si inexistant)")
    parser.add_argument("--tokenizer", type=str, default="../Tokenizer/tokenizer_model.bin")
    parser.add_argument("--cycles", type=int, default=2, 
                       help="Nombre de cycles d'entraînement")
    parser.add_argument("--articles", type=int, default=10, 
                       help="Articles Wikipedia par cycle")
    parser.add_argument("--qa-per-article", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=2, 
                       help="Epochs par cycle")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--language", type=str, default='fr', 
                       help="Langue Wikipedia (fr/en)")
    parser.add_argument("--device", type=str, 
                       default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    trainer = ContinuousTrainer(
        model_dir=args.model_dir,
        tokenizer_path=args.tokenizer,
        device=device,
        language=args.language
    )
    
    trainer.run_continuous_training(
        num_cycles=args.cycles,
        articles_per_cycle=args.articles,
        qa_per_article=args.qa_per_article,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )


if __name__ == "__main__":
    main()