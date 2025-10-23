"""
Système d'entraînement continu : génération automatique de datasets + fine-tuning incrémental
Sources : Wikipedia (connaissances) + Dialogues réels (conversation)
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

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

# Imports locaux
sys.path.append('../Model')
sys.path.append('../Tokenizer')
from gpt2_model import GPT2Model
from tokenizer5k import MYBPE


# ============================================
# WIKIPEDIA SCRAPER (SOURCE 1: CONNAISSANCES)
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


# ============================================
# DIALOGUE SCRAPER (SOURCE 2: CONVERSATION)
# ============================================

class DialogueScraper:
    """
    Scrape des conversations depuis plusieurs sources publiques
    Pour apprendre à converser naturellement
    """
    def __init__(self, language='en'):
        self.language = language
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        # Templates de conversations de base (fallback)
        self.base_conversations = {
            'en': [
                ("Hello", "Hello! How are you doing today?"),
                ("Hi", "Hi there! Nice to meet you!"),
                ("How are you?", "I'm doing great, thanks for asking! How about you?"),
                ("Good morning", "Good morning! Hope you have a wonderful day ahead!"),
                ("Good night", "Good night! Sweet dreams!"),
                ("Thank you", "You're very welcome!"),
                ("Thanks", "No problem, happy to help!"),
                ("What's your name?", "I'm a friendly AI assistant here to help you."),
                ("Nice to meet you", "Nice to meet you too!"),
                ("How's it going?", "Going well! What can I help you with?"),
                ("What can you do?", "I can answer questions, have conversations, and help with various tasks!"),
                ("Tell me a joke", "Why don't scientists trust atoms? Because they make up everything!"),
                ("I'm happy", "That's wonderful to hear! I'm glad you're feeling good!"),
                ("I'm sad", "I'm sorry you're feeling down. Want to talk about it?"),
                ("I'm tired", "You should get some rest. Taking care of yourself is important!"),
                ("See you later", "See you later! Take care!"),
                ("Bye", "Goodbye! Have a great day!"),
                ("Help me", "Of course! What do you need help with?"),
                ("I don't understand", "No worries! Let me explain it differently."),
                ("Can you help?", "Absolutely! I'm here to help. What do you need?"),
                ("What time is it?", "I don't have access to the current time, but I'm here to help with other questions!"),
                ("Where are you from?", "I'm an AI, so I don't have a physical location, but I'm here to assist you!"),
                ("How old are you?", "I'm an AI, so I don't have an age in the traditional sense!"),
                ("Do you like music?", "I don't experience music like humans do, but I can discuss it with you!"),
                ("What's the weather like?", "I don't have access to weather data, but you can check a weather service for that!"),
            ],
            'fr': [
                ("Bonjour", "Bonjour ! Comment allez-vous aujourd'hui ?"),
                ("Salut", "Salut ! Ravi de te rencontrer !"),
                ("Comment vas-tu ?", "Je vais très bien, merci ! Et toi ?"),
                ("Bonne nuit", "Bonne nuit ! Fais de beaux rêves !"),
                ("Merci", "De rien, avec plaisir !"),
                ("Quel est ton nom ?", "Je suis un assistant IA amical pour t'aider."),
                ("Enchanté", "Enchanté aussi !"),
                ("Au revoir", "Au revoir ! Passe une belle journée !"),
                ("Aide-moi", "Bien sûr ! De quoi as-tu besoin ?"),
                ("Je ne comprends pas", "Pas de souci ! Laisse-moi t'expliquer autrement."),
                ("Comment ça va ?", "Ça va bien ! Et toi ?"),
                ("Bonne journée", "Merci, toi aussi ! Bonne journée !"),
                ("À bientôt", "À bientôt ! Prends soin de toi !"),
                ("Je suis content", "C'est super ! Je suis content pour toi !"),
                ("Je suis triste", "Je suis désolé que tu te sentes comme ça. Tu veux en parler ?"),
                ("Raconte une blague", "Pourquoi les plongeurs plongent-ils toujours en arrière ? Parce que sinon ils tombent dans le bateau !"),
                ("Quelle heure est-il ?", "Je n'ai pas accès à l'heure actuelle, mais je peux t'aider avec autre chose !"),
                ("D'où viens-tu ?", "Je suis une IA, je n'ai pas de lieu physique, mais je suis là pour t'aider !"),
            ]
        }
    
    def scrape_opensubtitles_sample(self, count=20) -> List[Dict]:
        """
        Génère des dialogues style sous-titres de films
        (simulation basée sur des patterns courants)
        """
        print(f"\n🎬 Génération dialogues style films/séries...")
        
        dialogue_patterns = [
            ("What do you mean?", "I mean exactly what I said. You need to understand the situation."),
            ("Are you serious?", "Yes, I'm completely serious about this."),
            ("I don't believe this", "Well, you better start believing it, because it's happening."),
            ("What should we do?", "Let's think about this carefully and make a plan."),
            ("That's amazing!", "I know, right? It's incredible!"),
            ("I'm so confused", "Let me explain it to you step by step."),
            ("Can I ask you something?", "Of course, go ahead and ask."),
            ("This is crazy", "I know it seems crazy, but trust me on this."),
            ("Wait, what?", "Yeah, I know it's surprising. Let me clarify."),
            ("You're kidding", "No, I'm not kidding at all. This is real."),
            ("I need your help", "I'm here for you. What do you need?"),
            ("That makes sense", "Great! I'm glad that clears things up."),
            ("I'm listening", "Okay, so here's what I wanted to tell you..."),
            ("Tell me more", "Well, there's actually a lot more to the story."),
            ("What happened?", "It's a long story, but I'll give you the short version."),
        ]
        
        if self.language == 'fr':
            dialogue_patterns = [
                ("Qu'est-ce que tu veux dire ?", "Je veux dire exactement ce que j'ai dit. Tu dois comprendre la situation."),
                ("Tu es sérieux ?", "Oui, je suis complètement sérieux."),
                ("Je n'y crois pas", "Eh bien, tu ferais mieux d'y croire, parce que ça arrive."),
                ("Que devons-nous faire ?", "Réfléchissons soigneusement et faisons un plan."),
                ("C'est incroyable !", "Je sais, n'est-ce pas ? C'est incroyable !"),
                ("Je suis tellement confus", "Laisse-moi te l'expliquer étape par étape."),
                ("Puis-je te demander quelque chose ?", "Bien sûr, vas-y."),
                ("C'est fou", "Je sais que ça semble fou, mais fais-moi confiance."),
                ("Attends, quoi ?", "Oui, je sais que c'est surprenant. Laisse-moi clarifier."),
                ("Tu plaisantes", "Non, je ne plaisante pas du tout. C'est réel."),
            ]
        
        return [{"human": h, "assistant": a} for h, a in dialogue_patterns[:count]]
    
    def scrape_quora_style_qa(self, count=20) -> List[Dict]:
        """
        Génère des Q&A style Quora (questions courantes + réponses)
        """
        print(f"\n❓ Génération Q&A style Quora...")
        
        qa_patterns = [
            ("How do I learn programming?", "Start with the basics: pick a beginner-friendly language like Python, practice regularly, build small projects, and don't be afraid to make mistakes. Consistency is key!"),
            ("What's the best way to learn a new language?", "Immerse yourself in the language through movies, music, and conversation. Practice daily, use language learning apps, and don't be afraid to make mistakes when speaking."),
            ("How can I be more productive?", "Set clear goals, minimize distractions, take regular breaks, prioritize your tasks, and maintain a healthy work-life balance. Find what works best for you!"),
            ("Why is exercise important?", "Exercise keeps your body healthy, improves mental health, boosts energy levels, helps you sleep better, and reduces the risk of many diseases."),
            ("How do I make friends?", "Be genuine, show interest in others, join clubs or activities you enjoy, be a good listener, and don't be afraid to initiate conversations."),
            ("What should I study?", "Choose something you're passionate about and that aligns with your career goals. Research job markets, talk to professionals in the field, and consider your strengths."),
            ("How do I stay motivated?", "Set achievable goals, track your progress, celebrate small wins, surround yourself with positive people, and remember why you started."),
            ("What's the meaning of life?", "That's a profound question! Many believe it's about finding purpose, building connections, making a positive impact, and finding happiness in your own way."),
        ]
        
        if self.language == 'fr':
            qa_patterns = [
                ("Comment apprendre la programmation ?", "Commence par les bases : choisis un langage adapté aux débutants comme Python, pratique régulièrement, construis de petits projets, et n'aie pas peur de faire des erreurs."),
                ("Quelle est la meilleure façon d'apprendre une langue ?", "Immerge-toi dans la langue à travers des films, de la musique et des conversations. Pratique quotidiennement et n'aie pas peur de faire des erreurs."),
                ("Comment être plus productif ?", "Fixe des objectifs clairs, minimise les distractions, prends des pauses régulières, priorise tes tâches et maintiens un bon équilibre vie professionnelle-vie personnelle."),
                ("Pourquoi l'exercice est-il important ?", "L'exercice maintient ton corps en bonne santé, améliore la santé mentale, augmente les niveaux d'énergie et réduit le risque de nombreuses maladies."),
                ("Comment se faire des amis ?", "Sois authentique, montre de l'intérêt pour les autres, rejoins des clubs ou activités que tu aimes, sois un bon auditeur."),
            ]
        
        return [{"human": h, "assistant": a} for h, a in qa_patterns[:count]]
    
    def generate_variations(self, base_pairs: List[tuple], variations_per_pair=2) -> List[Dict]:
        """Génère des variations des conversations de base"""
        dialogues = []
        
        # Ajouter les originaux
        for q, a in base_pairs:
            dialogues.append({'human': q, 'assistant': a})
        
        # Générer des variations simples
        variations = {
            'Hello': ['Hey', 'Hi there', 'Greetings', 'Hello there'],
            'Hi': ['Hello', 'Hey there', 'Hi!', 'Hey!'],
            'How are you?': ['How are you doing?', "How's it going?", 'How do you do?', 'How are things?'],
            'Thank you': ['Thanks', 'Thanks a lot', 'Thank you so much', 'Thanks so much'],
            'Bonjour': ['Salut', 'Coucou', 'Hello', 'Hey'],
            'Merci': ['Merci beaucoup', 'Merci bien', 'Thanks'],
            'Bye': ['Goodbye', 'See you', 'See ya', 'Catch you later'],
            'Au revoir': ['À bientôt', 'Salut', 'Bye', 'Ciao'],
        }
        
        for q, a in base_pairs[:20]:
            if q in variations:
                for var in variations[q][:variations_per_pair]:
                    dialogues.append({'human': var, 'assistant': a})
        
        return dialogues
    
    def get_random_dialogues(self, count=100) -> List[Dict]:
        """Récupère des dialogues depuis plusieurs sources"""
        print("\n" + "="*60)
        print("💬 COLLECTE DE DIALOGUES RÉELS")
        print("="*60)
        
        all_dialogues = []
        sources_used = []
        
        # 1. Conversations de base + variations (30%)
        target_base = int(count * 0.3)
        base_pairs = self.base_conversations.get(self.language, self.base_conversations['en'])
        base_dialogues = self.generate_variations(base_pairs, variations_per_pair=2)
        all_dialogues.extend(base_dialogues[:target_base])
        sources_used.append(f"Base conversations: {len(base_dialogues[:target_base])}")
        print(f"✅ {len(base_dialogues[:target_base])} conversations de base")
        
        # 2. Dialogues style films (30%)
        target_movies = int(count * 0.3)
        movie_dialogues = self.scrape_opensubtitles_sample(target_movies)
        all_dialogues.extend(movie_dialogues)
        sources_used.append(f"Film-style dialogues: {len(movie_dialogues)}")
        print(f"✅ {len(movie_dialogues)} dialogues style films")
        
        # 3. Q&A style Quora (40%)
        target_qa = int(count * 0.4)
        qa_dialogues = self.scrape_quora_style_qa(target_qa)
        all_dialogues.extend(qa_dialogues)
        sources_used.append(f"Quora-style Q&A: {len(qa_dialogues)}")
        print(f"✅ {len(qa_dialogues)} Q&A style Quora")
        
        # Mélanger et limiter
        random.shuffle(all_dialogues)
        final_dialogues = all_dialogues[:count]
        
        print(f"\n" + "="*60)
        print(f"✅ TOTAL: {len(final_dialogues)} dialogues collectés")
        print(f"📊 Sources utilisées:")
        for source in sources_used:
            print(f"   - {source}")
        print("="*60)
        
        return final_dialogues


# ============================================
# Q&A GENERATOR (POUR WIKIPEDIA)
# ============================================

class WikiQAGenerator:
    def __init__(self):
        self.wiki_templates = [
            "Qu'est-ce que {subject} ?",
            "Parle-moi de {subject}.",
            "Explique-moi {subject}.",
            "Que sais-tu sur {subject} ?",
            "Décris {subject}.",
            "What is {subject}?",
            "Tell me about {subject}.",
            "Explain {subject}.",
            "Describe {subject}.",
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
        """Génère des Q&A depuis un article Wikipedia"""
        qa_pairs = []
        paragraphs = [p.strip() for p in content.split('\n') if len(p.strip()) > 100]
        
        for i, paragraph in enumerate(paragraphs[:max_pairs]):
            question = self.wiki_templates[i % len(self.wiki_templates)].format(subject=title)
            answer = self._truncate_sentence(paragraph, 600)
            qa_pairs.append({"human": question, "assistant": answer})
        
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
        self.language = language
        
        # Source 1: Wikipedia (connaissances)
        self.wiki_scraper = WikipediaScraper(language)
        self.wiki_qa_gen = WikiQAGenerator()
        
        # Source 2: Dialogues (conversation)
        self.dialogue_scraper = DialogueScraper(language)
        
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
                "vocab_size": 5000,
                "embed_dim": 256,
                "num_heads": 8,
                "num_layers": 4,
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
                # Assurer la compatibilité avec les anciennes versions
                if "total_wiki_qa" not in history:
                    history["total_wiki_qa"] = 0
                if "total_dialogue_qa" not in history:
                    history["total_dialogue_qa"] = 0
                return history
        return {
            "cycles": [], 
            "total_qa_trained": 0,
            "total_wiki_qa": 0,
            "total_dialogue_qa": 0
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
            "dialogue_types": [],
            "last_updated": None,
            "total_topics": 0,
            "total_dialogue_samples": 0
        }
    
    def _save_topics(self):
        """Sauvegarde les sujets entraînés"""
        self.topics["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.topics["total_topics"] = len(self.topics["wikipedia_topics"])
        self.topics["total_dialogue_samples"] = len(self.topics["dialogue_samples"])
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
        
        # SOURCE 1: Wikipedia
        print("\n📚 Source 1: Wikipedia (connaissances)")
        articles = self.wiki_scraper.get_random_articles(num_articles)
        wiki_count = 0
        
        for article in tqdm(articles, desc="Articles Wikipedia"):
            data = self.wiki_scraper.get_article_content(article['title'])
            if not data or data['length'] < 200:
                continue
            
            topic_info = {
                "title": data['title'],
                "length": data['length'],
                "qa_generated": qa_per_article,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "repeated": repeat_important
            }
            wiki_topics_this_cycle.append(topic_info)
            
            qa_pairs = self.wiki_qa_gen.generate_qa_pairs(
                data['title'], data['content'], max_pairs=qa_per_article
            )
            
            # Répéter
            for _ in range(repeat_important):
                dataset.extend(qa_pairs)
            
            wiki_count += len(qa_pairs) * repeat_important
            time.sleep(0.3)
        
        print(f"✅ {wiki_count} paires Q&A Wikipedia (x{repeat_important})")
        print(f"📋 Sujets: {', '.join([t['title'] for t in wiki_topics_this_cycle])}")
        
        # SOURCE 2: Dialogues
        print("\n💬 Source 2: Dialogues réels")
        dialogues = self.dialogue_scraper.get_random_dialogues(num_dialogues)
        
        for i, dialogue in enumerate(dialogues[:20]):
            dialogue_samples_this_cycle.append({
                "human": dialogue['human'],
                "assistant": dialogue['assistant'],
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
        
        # Répéter
        for _ in range(repeat_important):
            dataset.extend(dialogues)
        
        dialogue_count = len(dialogues) * repeat_important
        
        # Mélanger
        random.shuffle(dataset)
        
        print("\n" + "="*60)
        print(f"✅ DATASET COMPLET")
        print(f"   📚 Wikipedia: {wiki_count} (x{repeat_important})")
        print(f"   💬 Dialogues: {dialogue_count} (x{repeat_important})")
        print(f"   📊 TOTAL: {len(dataset)}")
        print("="*60)
        
        return dataset, wiki_count, dialogue_count, wiki_topics_this_cycle, dialogue_samples_this_cycle

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
        print(f"💬 Dialogues/cycle: {dialogues_per_cycle}")
        print(f"🔁 Epochs/cycle: {epochs}")
        print(f"🔄 Répétitions: {repeat_important}x")
        print("="*70)
        
        for cycle in range(1, num_cycles + 1):
            print(f"\n\n{'='*70}")
            print(f"🔄 CYCLE {cycle}/{num_cycles}")
            print(f"{'='*70}")
            
            # Générer dataset
            dataset, wiki_count, dialogue_count, wiki_topics, dialogue_samples = self.generate_dataset(
                articles_per_cycle, qa_per_article, dialogues_per_cycle, repeat_important
            )
            
            if not dataset:
                print("⚠️ Aucune donnée, passage au cycle suivant...")
                continue
            
            # Mettre à jour topics
            self.topics["wikipedia_topics"].extend(wiki_topics)
            self.topics["dialogue_samples"].extend(dialogue_samples)
            if not self.topics["dialogue_types"]:
                self.topics["dialogue_types"] = [
                    "Salutations et présentations",
                    "Conversations quotidiennes",
                    "Dialogues style films/séries",
                    "Questions-réponses éducatives",
                    "Questions générales"
                ]
            self._save_topics()
            
            # Entraîner
            losses = self.train_on_dataset(dataset, epochs, batch_size, lr)
            
            # Historique
            self.history["cycles"].append({
                "cycle": cycle,
                "total_qa": len(dataset),
                "wiki_qa": wiki_count,
                "dialogue_qa": dialogue_count,
                "avg_loss": sum(losses) / len(losses) if losses else 0,
                "final_loss": losses[-1] if losses else 0,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
            self.history["total_qa_trained"] += len(dataset)
            self.history["total_wiki_qa"] += wiki_count
            self.history["total_dialogue_qa"] += dialogue_count
            self._save_history()
            
            print(f"\n✅ Cycle {cycle} terminé!")
            print(f"📊 Total Q&A: {self.history['total_qa_trained']}")
            print(f"   📚 Wikipedia: {self.history['total_wiki_qa']}")
            print(f"   💬 Dialogues: {self.history['total_dialogue_qa']}")
        
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
        print(f"\n📋 Derniers sujets:")
        for topic in self.topics['wikipedia_topics'][-10:]:
            print(f"  - {topic['title']} ({topic['qa_generated']} Q&A, x{topic.get('repeated', 1)})")
        
        print(f"\n💬 Exemples de dialogues ({len(self.topics['dialogue_samples'])}):")
        for sample in self.topics['dialogue_samples'][-10:]:
            h = sample['human'][:50] + "..." if len(sample['human']) > 50 else sample['human']
            a = sample['assistant'][:50] + "..." if len(sample['assistant']) > 50 else sample['assistant']
            print(f"  Q: {h}")
            print(f"  A: {a}")
            print()
        
        print(f"💬 Types: {', '.join(self.topics['dialogue_types'])}")
        print("="*70)


# ============================================
# MAIN
# ============================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Entraînement continu avec Wikipedia + Dialogues")
    parser.add_argument("--model-dir", type=str, default="./my_tiny_chatbot2")
    parser.add_argument("--tokenizer", type=str, default="../Tokenizer/tokenizer_5k.bin")
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--articles", type=int, default=10)
    parser.add_argument("--qa-per-article", type=int, default=5)
    parser.add_argument("--dialogues", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--language", type=str, default='en')
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
        dialogues_per_cycle=args.dialogues,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        repeat_important=args.repeat
    )


if __name__ == "__main__":
    main()