"""
Système d'entraînement continu : génération automatique de datasets + fine-tuning incrémental
Sources : Wikipedia (connaissances catégorisées) + OASST1 de Hugging Face (conversation)
"""

import os
import sys
import json
import time
import requests
import re
from tqdm import tqdm
from typing import List, Dict
import random
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

# Imports locaux
sys.path.append('../Model')
sys.path.append('../Tokenizer')
from gpt2_model import GPT2Model
from tokenizer5k import MYBPE

# Hugging Face datasets
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    print("⚠️ 'datasets' non installé. Installez avec: pip install datasets")
    HF_AVAILABLE = False


# ============================================
# WIKIPEDIA SCRAPER AVEC CATÉGORISATION
# ============================================

class WikipediaScraper:
    def __init__(self, language='fr'):
        self.language = language
        self.api_url = f"https://{language}.wikipedia.org/w/api.php"
        self.headers = {"User-Agent": "WikiQABot/1.0"}
        
        # Mots-clés pour catégorisation
        self.category_keywords = {
            'science': ['science', 'physique', 'chimie', 'biologie', 'mathématiques', 
                       'physics', 'chemistry', 'biology', 'mathematics', 'scientific'],
            'histoire': ['histoire', 'historical', 'guerre', 'war', 'siècle', 'century',
                        'ancien', 'ancient', 'révolution', 'revolution'],
            'géographie': ['pays', 'country', 'ville', 'city', 'région', 'region',
                          'continent', 'géographie', 'geography', 'capitale', 'capital'],
            'technologie': ['technologie', 'technology', 'informatique', 'computer',
                           'internet', 'logiciel', 'software', 'programmation', 'programming'],
            'art': ['art', 'peinture', 'painting', 'musique', 'music', 'sculpture',
                   'artiste', 'artist', 'œuvre', 'artwork'],
            'sport': ['sport', 'football', 'basketball', 'athlète', 'athlete',
                     'compétition', 'competition', 'olympique', 'olympic'],
            'politique': ['politique', 'politics', 'gouvernement', 'government',
                         'président', 'president', 'élection', 'election'],
            'économie': ['économie', 'economy', 'finance', 'entreprise', 'company',
                        'marché', 'market', 'banque', 'bank'],
            'nature': ['nature', 'animal', 'plante', 'plant', 'écologie', 'ecology',
                      'environnement', 'environment', 'espèce', 'species'],
            'culture': ['culture', 'société', 'society', 'tradition', 'religion',
                       'philosophie', 'philosophy', 'littérature', 'literature']
        }

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
            
            # Catégorisation
            category = self._categorize_article(title, text)
            
            return {
                'title': title, 
                'content': text, 
                'length': len(text),
                'category': category
            }
        except:
            return None

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'==+ .*? ==+', '', text)
        text = re.sub(r'\n{2,}', '\n', text)
        text = re.sub(r'\s{2,}', ' ', text)
        return text.strip()
    
    def _categorize_article(self, title: str, content: str) -> str:
        """Catégorise un article selon ses mots-clés"""
        # Combiner titre et contenu (limité pour performance)
        text = (title + " " + content[:1000]).lower()
        
        # Compter les occurrences de mots-clés par catégorie
        category_scores = {}
        for category, keywords in self.category_keywords.items():
            score = sum(text.count(keyword.lower()) for keyword in keywords)
            category_scores[category] = score
        
        # Trouver la catégorie avec le plus haut score
        if max(category_scores.values()) > 0:
            best_category = max(category_scores, key=category_scores.get)
            return best_category
        
        return 'général'


# ============================================
# OASST1 DIALOGUE LOADER (HUGGING FACE)
# ============================================

class OASST1DialogueLoader:
    """
    Charge progressivement des conversations depuis le dataset OASST1 de Hugging Face
    """
    def __init__(self, language='en', batch_size=50):
        self.language = language
        self.batch_size = batch_size
        self.dataset = None
        self.current_index = 0
        self.total_available = 0
        
        if not HF_AVAILABLE:
            print("⚠️ Hugging Face datasets non disponible")
            return
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Charge le dataset OASST1"""
        print("\n📦 Chargement du dataset OASST1 depuis Hugging Face...")
        try:
            # Charger le dataset (train split)
            self.dataset = load_dataset("OpenAssistant/oasst1", split="train")
            self.total_available = len(self.dataset)
            print(f"✅ Dataset chargé : {self.total_available} conversations disponibles")
            
            # Filtrer par langue si nécessaire
            if self.language != 'en':
                print(f"🔍 Filtrage pour langue: {self.language}")
                self.dataset = self.dataset.filter(lambda x: x.get('lang', 'en') == self.language)
                print(f"✅ {len(self.dataset)} conversations en {self.language}")
                self.total_available = len(self.dataset)
        
        except Exception as e:
            print(f"❌ Erreur lors du chargement OASST1: {e}")
            self.dataset = None
    
    def get_next_batch(self, count=None) -> List[Dict]:
        """Récupère le prochain batch de conversations"""
        if self.dataset is None:
            return self._get_fallback_dialogues(count or self.batch_size)
        
        if count is None:
            count = self.batch_size
        
        # Si on a atteint la fin, recommencer
        if self.current_index >= self.total_available:
            print("🔄 Fin du dataset atteinte, retour au début")
            self.current_index = 0
        
        dialogues = []
        end_index = min(self.current_index + count, self.total_available)
        
        print(f"\n📥 Extraction conversations {self.current_index} à {end_index}")
        
        for i in range(self.current_index, end_index):
            item = self.dataset[i]
            
            # OASST1 structure: message_tree avec parent_id
            # On prend les paires prompt-response
            if item.get('role') == 'prompter' and item.get('text'):
                prompt = item['text']
                
                # Chercher la réponse (message avec parent_id = current message_id)
                message_id = item.get('message_id')
                if message_id:
                    # Dans OASST1, on peut avoir des réponses multiples
                    # Pour simplifier, on prend la première réponse trouvée
                    for j in range(i+1, min(i+10, self.total_available)):
                        potential_response = self.dataset[j]
                        if (potential_response.get('parent_id') == message_id and 
                            potential_response.get('role') == 'assistant'):
                            response = potential_response.get('text', '')
                            if response:
                                dialogues.append({
                                    'human': prompt.strip(),
                                    'assistant': response.strip()
                                })
                            break
        
        self.current_index = end_index
        
        if not dialogues:
            print("⚠️ Aucune conversation extraite, utilisation du fallback")
            return self._get_fallback_dialogues(count)
        
        print(f"✅ {len(dialogues)} conversations extraites")
        return dialogues
    
    def _get_fallback_dialogues(self, count) -> List[Dict]:
        """Dialogues de secours si OASST1 n'est pas disponible"""
        print("⚠️ Utilisation des dialogues de fallback")
        
        fallback = [
            ("Hello", "Hello! How can I help you today?"),
            ("Hi", "Hi there! What can I do for you?"),
            ("How are you?", "I'm doing well, thank you for asking! How are you?"),
            ("What's your name?", "I'm an AI assistant here to help you."),
            ("Tell me a joke", "Why don't scientists trust atoms? Because they make up everything!"),
            ("Help me", "Of course! What do you need help with?"),
            ("Thank you", "You're welcome! Happy to help!"),
            ("Goodbye", "Goodbye! Have a great day!"),
            ("What can you do?", "I can answer questions, have conversations, and help with various tasks!"),
            ("How does this work?", "I use machine learning to understand and respond to your questions."),
        ]
        
        if self.language == 'fr':
            fallback = [
                ("Bonjour", "Bonjour ! Comment puis-je vous aider aujourd'hui ?"),
                ("Salut", "Salut ! Que puis-je faire pour toi ?"),
                ("Comment vas-tu ?", "Je vais bien, merci ! Et toi ?"),
                ("Quel est ton nom ?", "Je suis un assistant IA ici pour t'aider."),
                ("Raconte une blague", "Pourquoi les plongeurs plongent-ils en arrière ? Parce que sinon ils tombent dans le bateau !"),
                ("Aide-moi", "Bien sûr ! De quoi as-tu besoin ?"),
                ("Merci", "De rien ! Ravi d'avoir pu aider !"),
                ("Au revoir", "Au revoir ! Passe une excellente journée !"),
            ]
        
        return [{"human": h, "assistant": a} for h, a in fallback[:count]]
    
    def get_stats(self) -> Dict:
        """Statistiques sur le dataset"""
        return {
            "total_available": self.total_available,
            "current_index": self.current_index,
            "progress": f"{self.current_index}/{self.total_available}",
            "percentage": f"{(self.current_index/self.total_available*100):.1f}%" if self.total_available > 0 else "0%"
        }


# ============================================
# Q&A GENERATOR AMÉLIORÉ (AVEC CATÉGORIES)
# ============================================

class WikiQAGenerator:
    def __init__(self):
        # Templates par catégorie
        self.category_templates = {
            'science': [
                "Qu'est-ce que {subject} en science ?",
                "Explique-moi les concepts scientifiques de {subject}.",
                "Comment fonctionne {subject} scientifiquement ?",
                "What is {subject} in science?",
                "Explain the scientific concepts of {subject}.",
            ],
            'histoire': [
                "Quelle est l'histoire de {subject} ?",
                "Parle-moi des événements historiques de {subject}.",
                "Quand {subject} a-t-il eu lieu ?",
                "What is the history of {subject}?",
                "Tell me about the historical events of {subject}.",
            ],
            'géographie': [
                "Où se trouve {subject} ?",
                "Parle-moi de la géographie de {subject}.",
                "Quelles sont les caractéristiques géographiques de {subject} ?",
                "Where is {subject} located?",
                "Tell me about the geography of {subject}.",
            ],
            'technologie': [
                "Comment fonctionne la technologie de {subject} ?",
                "Explique-moi {subject} technologiquement.",
                "Quelles sont les applications de {subject} ?",
                "How does {subject} technology work?",
                "Explain {subject} technologically.",
            ],
            'général': [
                "Qu'est-ce que {subject} ?",
                "Parle-moi de {subject}.",
                "Explique-moi {subject}.",
                "What is {subject}?",
                "Tell me about {subject}.",
            ]
        }

    def _truncate_sentence(self, text: str, max_len=500):
        if len(text) <= max_len:
            return text.strip()
        truncated = text[:max_len]
        end = max(truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?'))
        if end != -1:
            truncated = truncated[:end + 1]
        return truncated.strip()

    def generate_qa_pairs(self, title: str, content: str, category: str, max_pairs=3) -> List[Dict]:
        """Génère des Q&A depuis un article Wikipedia avec questions adaptées à la catégorie"""
        qa_pairs = []
        paragraphs = [p.strip() for p in content.split('\n') if len(p.strip()) > 100]
        
        # Sélectionner les templates appropriés
        templates = self.category_templates.get(category, self.category_templates['général'])
        
        for i, paragraph in enumerate(paragraphs[:max_pairs]):
            question = templates[i % len(templates)].format(subject=title)
            answer = self._truncate_sentence(paragraph, 600)
            qa_pairs.append({
                "human": question, 
                "assistant": answer,
                "category": category
            })
        
        return qa_pairs


# ============================================
# DATASET + TRAINING (inchangé)
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
# CONTINUOUS TRAINING SYSTEM AMÉLIORÉ
# ============================================

class ContinuousTrainer:
    def __init__(self, model_dir, tokenizer_path, device, language='fr'):
        self.model_dir = model_dir
        self.tokenizer_path = tokenizer_path
        self.device = device
        self.language = language
        
        # Source 1: Wikipedia (connaissances catégorisées)
        self.wiki_scraper = WikipediaScraper(language)
        self.wiki_qa_gen = WikiQAGenerator()
        
        # Source 2: OASST1 (dialogues réels de HF)
        self.dialogue_loader = OASST1DialogueLoader(language, batch_size=50)
        
        # Créer le dossier si nécessaire
        os.makedirs(model_dir, exist_ok=True)
        
        # Charger ou initialiser le modèle
        self.model, self.tokenizer, self.config = self._load_or_init_model()
        
        # Historique de l'entraînement
        self.history_file = os.path.join(model_dir, "training_history.json")
        self.history = self._load_history()
        
        # Fichier des sujets entraînés
        self.topics_file = os.path.join(model_dir, "trained_topics.json")
        self.topics = self._load_topics()

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
                history = json.load(f)
                if "total_wiki_qa" not in history:
                    history["total_wiki_qa"] = 0
                if "total_dialogue_qa" not in history:
                    history["total_dialogue_qa"] = 0
                if "categories_trained" not in history:
                    history["categories_trained"] = {}
                return history
        return {
            "cycles": [], 
            "total_qa_trained": 0,
            "total_wiki_qa": 0,
            "total_dialogue_qa": 0,
            "categories_trained": {}
        }

    def _save_history(self):
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def _load_topics(self):
        """Charge les sujets déjà entraînés"""
        if os.path.exists(self.topics_file):
            with open(self.topics_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            "wikipedia_topics": [],
            "dialogue_samples": [],
            "categories": {},
            "oasst1_progress": {"current_index": 0, "total": 0},
            "last_updated": None,
            "total_topics": 0,
            "total_dialogue_samples": 0
        }
    
    def _save_topics(self):
        """Sauvegarde les sujets entraînés"""
        self.topics["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.topics["total_topics"] = len(self.topics["wikipedia_topics"])
        self.topics["total_dialogue_samples"] = len(self.topics["dialogue_samples"])
        
        # Stats OASST1
        if self.dialogue_loader.dataset:
            self.topics["oasst1_progress"] = self.dialogue_loader.get_stats()
        
        with open(self.topics_file, 'w', encoding='utf-8') as f:
            json.dump(self.topics, f, indent=2, ensure_ascii=False)

    def generate_dataset(self, num_articles=10, qa_per_article=3, num_dialogues=50, repeat_important=3):
        """
        Génère un dataset mixte depuis 2 sources avec répétitions
        """
        print("\n" + "="*60)
        print("🔄 GÉNÉRATION DATASET MIXTE")
        print("="*60)
        
        dataset = []
        wiki_topics_this_cycle = []
        dialogue_samples_this_cycle = []
        categories_count = {}
        
        # SOURCE 1: Wikipedia avec catégorisation
        print("\n📚 Source 1: Wikipedia (connaissances catégorisées)")
        articles = self.wiki_scraper.get_random_articles(num_articles)
        wiki_count = 0
        
        for article in tqdm(articles, desc="Articles Wikipedia"):
            data = self.wiki_scraper.get_article_content(article['title'])
            if not data or data['length'] < 200:
                continue
            
            category = data['category']
            categories_count[category] = categories_count.get(category, 0) + 1
            
            topic_info = {
                "title": data['title'],
                "length": data['length'],
                "category": category,
                "qa_generated": qa_per_article,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "repeated": repeat_important
            }
            wiki_topics_this_cycle.append(topic_info)
            
            qa_pairs = self.wiki_qa_gen.generate_qa_pairs(
                data['title'], data['content'], category, max_pairs=qa_per_article
            )
            
            # Répéter
            for _ in range(repeat_important):
                dataset.extend(qa_pairs)
            
            wiki_count += len(qa_pairs) * repeat_important
            time.sleep(0.3)
        
        print(f"✅ {wiki_count} paires Q&A Wikipedia (x{repeat_important})")
        print(f"📋 Catégories: {categories_count}")
        
        # SOURCE 2: OASST1 Dialogues
        print("\n💬 Source 2: OASST1 (Hugging Face)")
        dialogues = self.dialogue_loader.get_next_batch(num_dialogues)
        
        for i, dialogue in enumerate(dialogues[:20]):
            dialogue_samples_this_cycle.append({
                "human": dialogue['human'][:100] + "..." if len(dialogue['human']) > 100 else dialogue['human'],
                "assistant": dialogue['assistant'][:100] + "..." if len(dialogue['assistant']) > 100 else dialogue['assistant'],
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
        
        # Répéter
        for _ in range(repeat_important):
            dataset.extend(dialogues)
        
        dialogue_count = len(dialogues) * repeat_important
        
        # Stats OASST1
        oasst1_stats = self.dialogue_loader.get_stats()
        print(f"✅ {dialogue_count} dialogues OASST1 (x{repeat_important})")
        print(f"📊 Progression OASST1: {oasst1_stats['progress']} ({oasst1_stats['percentage']})")
        
        # Mélanger
        random.shuffle(dataset)
        
        print("\n" + "="*60)
        print(f"✅ DATASET COMPLET")
        print(f"   📚 Wikipedia: {wiki_count} (x{repeat_important})")
        print(f"   💬 OASST1: {dialogue_count} (x{repeat_important})")
        print(f"   📊 TOTAL: {len(dataset)}")
        print("="*60)
        
        return dataset, wiki_count, dialogue_count, wiki_topics_this_cycle, dialogue_samples_this_cycle, categories_count

    def train_on_dataset(self, dataset, epochs=2, batch_size=8, lr=5e-5):
        """Entraîne le modèle"""
        print("\n" + "="*60)
        print("🚀 ENTRAÎNEMENT")
        print("="*60)
        
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
        
        # Sauvegarder
        model_path = os.path.join(self.model_dir, "model.pt")
        torch.save(self.model.state_dict(), model_path)
        print(f"💾 Modèle sauvegardé : {model_path}")
        
        return cycle_losses

    def run_continuous_training(self, num_cycles=5, articles_per_cycle=10, 
                                qa_per_article=3, dialogues_per_cycle=50,
                                epochs=2, batch_size=8, lr=5e-5, repeat_important=3):
        """Boucle d'entraînement continu"""
        print("\n" + "="*70)
        print("🤖 ENTRAÎNEMENT CONTINU - DÉMARRAGE")
        print("="*70)
        print(f"📊 Cycles: {num_cycles}")
        print(f"📚 Articles/cycle: {articles_per_cycle}")
        print(f"💬 Dialogues OASST1/cycle: {dialogues_per_cycle}")
        print(f"🔁 Epochs/cycle: {epochs}")
        print(f"🔄 Répétitions: {repeat_important}x")
        print("="*70)
        
        for cycle in range(1, num_cycles + 1):
            print(f"\n\n{'='*70}")
            print(f"🔄 CYCLE {cycle}/{num_cycles}")
            print(f"{'='*70}")
            
            # Générer dataset
            dataset, wiki_count, dialogue_count, wiki_topics, dialogue_samples, categories = self.generate_dataset(
                articles_per_cycle, qa_per_article, dialogues_per_cycle, repeat_important
            )
            
            if not dataset:
                print("⚠️ Aucune donnée, passage au cycle suivant...")
                continue
            
            # Mettre à jour topics et catégories
            self.topics["wikipedia_topics"].extend(wiki_topics)
            self.topics["dialogue_samples"].extend(dialogue_samples)
            
            # Mise à jour des statistiques de catégories
            for category, count in categories.items():
                if category not in self.topics["categories"]:
                    self.topics["categories"][category] = 0
                self.topics["categories"][category] += count
            
            # Mise à jour historique catégories
            for category, count in categories.items():
                if category not in self.history["categories_trained"]:
                    self.history["categories_trained"][category] = 0
                self.history["categories_trained"][category] += count
            
            self._save_topics()
            
            # Entraîner
            losses = self.train_on_dataset(dataset, epochs, batch_size, lr)
            
            # Historique
            self.history["cycles"].append({
                "cycle": cycle,
                "total_qa": len(dataset),
                "wiki_qa": wiki_count,
                "dialogue_qa": dialogue_count,
                "categories": categories,
                "avg_loss": sum(losses) / len(losses) if losses else 0,
                "final_loss": losses[-1] if losses else 0,
                "oasst1_progress": self.dialogue_loader.get_stats(),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
            self.history["total_qa_trained"] += len(dataset)
            self.history["total_wiki_qa"] += wiki_count
            self.history["total_dialogue_qa"] += dialogue_count
            self._save_history()
            
            print(f"\n✅ Cycle {cycle} terminé!")
            print(f"📊 Total Q&A: {self.history['total_qa_trained']}")
            print(f"   📚 Wikipedia: {self.history['total_wiki_qa']}")
            print(f"   💬 OASST1: {self.history['total_dialogue_qa']}")
            print(f"   🏷️ Catégories: {self.history['categories_trained']}")
        
        print("\n" + "="*70)
        print("🎉 ENTRAÎNEMENT TERMINÉ!")
        print(f"📊 {num_cycles} cycles complétés")
        print(f"💾 Modèle: {os.path.join(self.model_dir, 'model.pt')}")
        print(f"📋 Topics: {self.topics_file}")
        print("="*70)
        
        self._print_topics_summary()
    
    def _print_topics_summary(self):
        """Résumé des sujets"""
        print("\n" + "="*70)
        print("📚 RÉSUMÉ DES SUJETS ENTRAÎNÉS")
        print("="*70)
        print(f"Total sujets Wikipedia: {len(self.topics['wikipedia_topics'])}")
        
        print(f"\n🏷️ Catégories Wikipedia:")
        for category, count in sorted(self.topics['categories'].items(), key=lambda x: x[1], reverse=True):
            print(f"  - {category}: {count} articles")
        
        print(f"\n📋 Derniers sujets par catégorie:")
        # Grouper par catégorie
        by_category = {}
        for topic in self.topics['wikipedia_topics'][-30:]:
            cat = topic.get('category', 'général')
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(topic)
        
        for category, topics in by_category.items():
            print(f"\n  {category.upper()}:")
            for topic in topics[:5]:
                print(f"    - {topic['title']} ({topic['qa_generated']} Q&A, x{topic.get('repeated', 1)})")
        
        print(f"\n💬 Dialogues OASST1:")
        print(f"  Total échantillons: {len(self.topics['dialogue_samples'])}")
        if 'oasst1_progress' in self.topics:
            stats = self.topics['oasst1_progress']
            print(f"  Progression dataset: {stats.get('progress', 'N/A')} ({stats.get('percentage', 'N/A')})")
        
        print(f"\n📝 Exemples de dialogues récents:")
        for sample in self.topics['dialogue_samples'][-5:]:
            print(f"  Q: {sample['human']}")
            print(f"  A: {sample['assistant']}")
            print()
        
        print("="*70)


# ============================================
# MAIN
# ============================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Entraînement continu avec Wikipedia catégorisé + OASST1")
    parser.add_argument("--model-dir", type=str, default="./my_tiny_chatbot_v2")
    parser.add_argument("--tokenizer", type=str, default="../Tokenizer/tokenizer_5k.bin")
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--articles", type=int, default=10)
    parser.add_argument("--qa-per-article", type=int, default=5)
    parser.add_argument("--dialogues", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--language", type=str, default='en', 
                       help="Language for Wikipedia and OASST1 (en, fr, de, es, etc.)")
    parser.add_argument("--device", type=str, 
                       default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
    # Vérifier si HF datasets est disponible
    if not HF_AVAILABLE:
        print("\n" + "="*70)
        print("⚠️ ATTENTION: Hugging Face 'datasets' n'est pas installé!")
        print("Pour utiliser OASST1, installez-le avec:")
        print("   pip install datasets")
        print("\nLe système utilisera des dialogues de fallback.")
        print("="*70)
        time.sleep(3)
    
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
        dialogues_per_cycle=args.dialogues,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        repeat_important=args.repeat
    )


if __name__ == "__main__":
    main()