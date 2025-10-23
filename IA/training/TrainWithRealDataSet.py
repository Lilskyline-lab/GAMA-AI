"""
Système d'entraînement avec Instruction Tuning intégré
Sources : Wikipedia (connaissances catégorisées) + OASST1 (conversation) + Fichiers personnalisés
Utilise le système d'instruction tuning pour formater automatiquement les données
"""

import os
import sys
import json
import time
import requests
import re
from tqdm import tqdm
from typing import List, Dict, Optional
import random
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model.gpt2_model import GPT2Model
from Tokenizer.tokenizer5k import MYBPE
from utils.instruction_tuning import (
    InstructionTemplates,
    InstructionTuningPipeline,
    convert_to_instruction_format,
    InstructionDatasetLoader
)

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    print("⚠️ 'datasets' non installé. Installez avec: pip install datasets")
    HF_AVAILABLE = False


class WikipediaScraper:
    def __init__(self, language='fr'):
        self.language = language
        self.api_url = f"https://{language}.wikipedia.org/w/api.php"
        self.headers = {"User-Agent": "WikiQABot/1.0"}
        
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
        text = (title + " " + content[:1000]).lower()
        category_scores = {}
        for category, keywords in self.category_keywords.items():
            score = sum(text.count(keyword.lower()) for keyword in keywords)
            category_scores[category] = score
        
        if max(category_scores.values()) > 0:
            best_category = max(category_scores, key=category_scores.get)
            return best_category
        return 'général'


class OASST1DialogueLoader:
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
        print("\n📦 Chargement du dataset OASST1 depuis Hugging Face...")
        try:
            self.dataset = load_dataset("OpenAssistant/oasst1", split="train")
            self.total_available = len(self.dataset)
            print(f"✅ Dataset chargé : {self.total_available} conversations disponibles")
            
            if self.language != 'en':
                print(f"🔍 Filtrage pour langue: {self.language}")
                self.dataset = self.dataset.filter(lambda x: x.get('lang', 'en') == self.language)
                print(f"✅ {len(self.dataset)} conversations en {self.language}")
                self.total_available = len(self.dataset)
        
        except Exception as e:
            print(f"❌ Erreur lors du chargement OASST1: {e}")
            self.dataset = None
    
    def get_next_batch(self, count=None) -> List[Dict]:
        if self.dataset is None:
            return self._get_fallback_dialogues(count or self.batch_size)
        
        if count is None:
            count = self.batch_size
        
        if self.current_index >= self.total_available:
            print("🔄 Fin du dataset atteinte, retour au début")
            self.current_index = 0
        
        dialogues = []
        end_index = min(self.current_index + count, self.total_available)
        
        print(f"\n📥 Extraction conversations {self.current_index} à {end_index}")
        
        for i in range(self.current_index, end_index):
            item = self.dataset[i]
            
            if item.get('role') == 'prompter' and item.get('text'):
                prompt = item['text']
                message_id = item.get('message_id')
                if message_id:
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
        return {
            "total_available": self.total_available,
            "current_index": self.current_index,
            "progress": f"{self.current_index}/{self.total_available}",
            "percentage": f"{(self.current_index/self.total_available*100):.1f}%" if self.total_available > 0 else "0%"
        }


class WikiQAGenerator:
    def __init__(self):
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
        qa_pairs = []
        paragraphs = [p.strip() for p in content.split('\n') if len(p.strip()) > 100]
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


class InstructionTunedDataset(Dataset):
    """
    Dataset qui applique automatiquement l'instruction tuning
    """
    def __init__(self, pairs, tokenizer, max_length=512, instruction_template="chat_bot"):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruction_template = instruction_template
        
        print(f"\n🎯 Application de l'instruction tuning (template: {instruction_template})")
        self.formatted_pairs = convert_to_instruction_format(
            pairs,
            template_name=instruction_template
        )
        print(f"✅ {len(self.formatted_pairs)} exemples formatés")

    def __len__(self):
        return len(self.formatted_pairs)

    def __getitem__(self, idx):
        formatted_text = self.formatted_pairs[idx]['formatted_text']
        h = self.pairs[idx]['human'].strip()
        a = self.pairs[idx]['assistant'].strip()
        
        prefix = f"Human: {h}\nBot:"
        
        ids_prefix = self.tokenizer.encoder(prefix)
        ids_all = self.tokenizer.encoder(formatted_text)
        
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


class ContinuousTrainer:
    def __init__(
        self,
        model_dir,
        tokenizer_path,
        device,
        language='fr',
        instruction_template="chat_bot",
        custom_data_dir=None
    ):
        self.model_dir = model_dir
        self.tokenizer_path = tokenizer_path
        self.device = device
        self.language = language
        self.instruction_template = instruction_template
        self.custom_data_dir = custom_data_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data'
        )
        
        self.wiki_scraper = WikipediaScraper(language)
        self.wiki_qa_gen = WikiQAGenerator()
        self.dialogue_loader = OASST1DialogueLoader(language, batch_size=50)
        
        os.makedirs(model_dir, exist_ok=True)
        
        self.model, self.tokenizer, self.config = self._load_or_init_model()
        
        self.history_file = os.path.join(model_dir, "training_history.json")
        self.history = self._load_history()
        
        self.topics_file = os.path.join(model_dir, "trained_topics.json")
        self.topics = self._load_topics()
        
        print(f"\n✅ Instruction Tuning activé (template: {instruction_template})")

    def _load_or_init_model(self):
        cfg_path = os.path.join(self.model_dir, "config.json")
        model_path = os.path.join(self.model_dir, "model.pt")
        
        if os.path.exists(cfg_path):
            with open(cfg_path, 'r') as f:
                cfg = json.load(f)
        else:
            cfg = {
                "vocab_size": 5000,
                "embed_dim": 256,
                "num_heads": 8,
                "num_layers": 4,
                "max_seq_len": 512
            }
            with open(cfg_path, 'w') as f:
                json.dump(cfg, f, indent=2)
        
        tokenizer = MYBPE(vocab_size=cfg["vocab_size"])
        tokenizer.load_tokenizer(self.tokenizer_path)
        
        model = GPT2Model(
            vocab_size=cfg["vocab_size"],
            embed_dim=cfg["embed_dim"],
            num_heads=cfg["num_heads"],
            num_layers=cfg["num_layers"],
            max_seq_len=cfg["max_seq_len"]
        )
        
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
            "categories_trained": {},
            "instruction_template_used": self.instruction_template
        }

    def _save_history(self):
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def _load_topics(self):
        if os.path.exists(self.topics_file):
            with open(self.topics_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            "wikipedia_topics": [],
            "dialogue_samples": [],
            "categories": {},
            "oasst1_progress": {"current_index": 0, "total": 0},
            "custom_files_used": [],
            "last_updated": None,
            "total_topics": 0,
            "total_dialogue_samples": 0
        }
    
    def _save_topics(self):
        self.topics["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.topics["total_topics"] = len(self.topics["wikipedia_topics"])
        self.topics["total_dialogue_samples"] = len(self.topics["dialogue_samples"])
        
        if self.dialogue_loader.dataset:
            self.topics["oasst1_progress"] = self.dialogue_loader.get_stats()
        
        with open(self.topics_file, 'w', encoding='utf-8') as f:
            json.dump(self.topics, f, indent=2, ensure_ascii=False)

    def load_custom_data(self) -> List[Dict]:
        """
        Charge les données personnalisées depuis le dossier data/
        """
        custom_data = []
        
        if not os.path.exists(self.custom_data_dir):
            print(f"⚠️ Dossier de données personnalisées non trouvé: {self.custom_data_dir}")
            return custom_data
        
        print(f"\n📂 Recherche de fichiers personnalisés dans: {self.custom_data_dir}")
        
        for filename in os.listdir(self.custom_data_dir):
            if filename.endswith(('.json', '.jsonl', '.csv')):
                filepath = os.path.join(self.custom_data_dir, filename)
                try:
                    data = InstructionDatasetLoader.load_dataset(filepath)
                    custom_data.extend(data)
                    print(f"✅ {len(data)} exemples chargés depuis {filename}")
                    
                    if filename not in self.topics.get("custom_files_used", []):
                        if "custom_files_used" not in self.topics:
                            self.topics["custom_files_used"] = []
                        self.topics["custom_files_used"].append(filename)
                
                except Exception as e:
                    print(f"⚠️ Erreur lors du chargement de {filename}: {e}")
        
        if custom_data:
            print(f"✅ Total: {len(custom_data)} exemples personnalisés chargés")
        else:
            print("ℹ️ Aucune donnée personnalisée trouvée")
        
        return custom_data

    def generate_dataset(
        self,
        num_articles=10,
        qa_per_article=3,
        num_dialogues=50,
        repeat_important=3,
        use_custom_data=True
    ):
        print("\n" + "="*60)
        print("🔄 GÉNÉRATION DATASET MIXTE AVEC INSTRUCTION TUNING")
        print("="*60)
        
        dataset = []
        wiki_topics_this_cycle = []
        dialogue_samples_this_cycle = []
        categories_count = {}
        
        # SOURCE 1: Données personnalisées
        if use_custom_data:
            custom_data = self.load_custom_data()
            if custom_data:
                dataset.extend(custom_data * repeat_important)
                print(f"\n📝 {len(custom_data)} exemples personnalisés ajoutés (×{repeat_important} répétitions)")
        
        # SOURCE 2: Wikipedia avec catégorisation
        print("\n📚 Source 2: Wikipedia (connaissances catégorisées)")
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
            
            for qa in qa_pairs:
                for _ in range(repeat_important):
                    dataset.append(qa)
                    wiki_count += 1
        
        self.topics["wikipedia_topics"].extend(wiki_topics_this_cycle)
        for cat, count in categories_count.items():
            if cat not in self.topics["categories"]:
                self.topics["categories"][cat] = 0
            self.topics["categories"][cat] += count
        
        print(f"\n✅ Wikipedia: {wiki_count} paires Q&A générées")
        print(f"📊 Catégories: {dict(categories_count)}")
        
        # SOURCE 3: OASST1 (dialogues)
        print("\n💬 Source 3: OASST1 (dialogues conversationnels)")
        dialogues = self.dialogue_loader.get_next_batch(num_dialogues)
        dialogue_count = len(dialogues)
        
        for dialogue in dialogues:
            dialogue_samples_this_cycle.append({
                "human_preview": dialogue['human'][:50] + "...",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
            
            for _ in range(repeat_important):
                dataset.append(dialogue)
        
        self.topics["dialogue_samples"].extend(dialogue_samples_this_cycle)
        
        print(f"✅ OASST1: {dialogue_count} dialogues extraits (×{repeat_important} répétitions)")
        
        # Mélanger
        random.shuffle(dataset)
        
        print(f"\n" + "="*60)
        print(f"✅ DATASET TOTAL: {len(dataset)} exemples")
        print(f"   - Données personnalisées: {len(custom_data) * repeat_important if use_custom_data and custom_data else 0}")
        print(f"   - Wikipedia Q&A: {wiki_count}")
        print(f"   - OASST1 dialogues: {dialogue_count * repeat_important}")
        print(f"🎯 Instruction Template: {self.instruction_template}")
        print("="*60)
        
        self._save_topics()
        
        return dataset

    def train_one_cycle(
        self,
        num_articles=10,
        qa_per_article=3,
        num_dialogues=50,
        epochs=3,
        batch_size=8,
        lr=5e-5,
        use_custom_data=True
    ):
        print("\n" + "="*70)
        print("🚀 DÉMARRAGE CYCLE D'ENTRAÎNEMENT AVEC INSTRUCTION TUNING")
        print("="*70)
        
        dataset_pairs = self.generate_dataset(
            num_articles, qa_per_article, num_dialogues, repeat_important=3,
            use_custom_data=use_custom_data
        )
        
        if not dataset_pairs:
            print("❌ Dataset vide, abandon du cycle")
            return
        
        dataset = InstructionTunedDataset(
            dataset_pairs,
            self.tokenizer,
            max_length=self.config["max_seq_len"],
            instruction_template=self.instruction_template
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, pad_id=0)
        )
        
        optimizer = AdamW(self.model.parameters(), lr=lr)
        loss_fn = CrossEntropyLoss(ignore_index=-100)
        
        self.model.train()
        
        total_loss = 0
        step = 0
        
        print(f"\n⏳ Entraînement sur {len(dataset)} exemples, {epochs} époques")
        
        for epoch in range(epochs):
            epoch_loss = 0
            pbar = tqdm(dataloader, desc=f"Époque {epoch+1}/{epochs}")
            
            for batch in pbar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                logits, _ = self.model(input_ids)
                
                loss = loss_fn(
                    logits.view(-1, self.config["vocab_size"]),
                    labels.view(-1)
                )
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                epoch_loss += loss.item()
                step += 1
                
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            avg_epoch_loss = epoch_loss / len(dataloader)
            print(f"✓ Époque {epoch+1} terminée - Loss moyenne: {avg_epoch_loss:.4f}")
        
        avg_loss = total_loss / step
        
        torch.save(self.model.state_dict(), os.path.join(self.model_dir, "model.pt"))
        print(f"\n✅ Modèle sauvegardé: {self.model_dir}/model.pt")
        
        cycle_info = {
            "cycle": len(self.history["cycles"]) + 1,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "examples": len(dataset),
            "epochs": epochs,
            "avg_loss": avg_loss,
            "instruction_template": self.instruction_template
        }
        self.history["cycles"].append(cycle_info)
        self.history["total_qa_trained"] += len(dataset)
        self._save_history()
        
        print("\n" + "="*70)
        print(f"✅ CYCLE {cycle_info['cycle']} TERMINÉ")
        print(f"   Loss moyenne: {avg_loss:.4f}")
        print(f"   Total exemples entraînés (historique): {self.history['total_qa_trained']}")
        print("="*70)

    def display_stats(self):
        print("\n" + "="*60)
        print("📊 STATISTIQUES D'ENTRAÎNEMENT")
        print("="*60)
        
        print(f"\n🔢 Cycles d'entraînement: {len(self.history['cycles'])}")
        print(f"📝 Total exemples entraînés: {self.history['total_qa_trained']}")
        print(f"🎯 Template d'instruction: {self.instruction_template}")
        
        if self.history['cycles']:
            last_cycle = self.history['cycles'][-1]
            print(f"\n🕐 Dernier cycle:")
            print(f"   - Date: {last_cycle['timestamp']}")
            print(f"   - Exemples: {last_cycle['examples']}")
            print(f"   - Loss: {last_cycle['avg_loss']:.4f}")
        
        print(f"\n📚 Sujets Wikipedia traités: {len(self.topics['wikipedia_topics'])}")
        
        if self.topics.get('categories'):
            print("\n📁 Distribution par catégories:")
            for cat, count in sorted(self.topics['categories'].items(), key=lambda x: x[1], reverse=True):
                print(f"   - {cat}: {count} articles")
        
        print(f"\n💬 Échantillons de dialogue: {len(self.topics['dialogue_samples'])}")
        
        if self.topics.get('custom_files_used'):
            print(f"\n📂 Fichiers personnalisés utilisés:")
            for filename in self.topics['custom_files_used']:
                print(f"   - {filename}")
        
        if 'oasst1_progress' in self.topics:
            stats = self.topics['oasst1_progress']
            print(f"\n📦 Progression OASST1: {stats.get('progress', 'N/A')}")
        
        print("="*60)


def main():
    print("\n" + "="*70)
    print("🤖 SYSTÈME D'ENTRAÎNEMENT LLM AVEC INSTRUCTION TUNING")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Device: {device}")
    
    model_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "saved_models",
        "my_llm"
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
    
    print(f"📁 Model directory: {model_dir}")
    print(f"🔤 Tokenizer: {tokenizer_path}")
    
    trainer = ContinuousTrainer(
        model_dir=model_dir,
        tokenizer_path=tokenizer_path,
        device=device,
        language='en',
        instruction_template="chat_bot"
    )
    
    trainer.display_stats()
    
    print("\n🎯 Démarrage de l'entraînement...")
    print("   - 10 articles Wikipedia (English)")
    print("   - 100 dialogues OASST1 (English)")
    print("   - 3 époques")
    print("   - Vocab: 5,000 tokens")
    print("   - Template: chat_bot (Human/Bot)")
    
    trainer.train_one_cycle(
        num_articles=10,
        qa_per_article=3,
        num_dialogues=100,
        epochs=3,
        batch_size=4,
        lr=5e-5,
        use_custom_data=False
    )
    
    trainer.display_stats()
    
    print("\n✅ Entraînement terminé!")
    print("💡 Les données sont automatiquement formatées avec instruction tuning")
    print("📊 Sources: Wikipedia + OASST1 (Hugging Face)")


if __name__ == "__main__":
    main()
