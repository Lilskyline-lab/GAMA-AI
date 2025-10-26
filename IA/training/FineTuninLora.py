"""
Système d'entraînement avec LoRA (Low-Rank Adaptation) + Instruction Tuning
Sources : Wikipedia (connaissances catégorisées) + OASST1 (conversation) + Fichiers personnalisés
Utilise LoRA pour un fine-tuning efficace avec peu de paramètres
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
import torch.nn as nn
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


# ============================================================================
# IMPLÉMENTATION LoRA
# ============================================================================

class LoRALayer(nn.Module):
    """
    Couche LoRA pour adapter une couche linéaire existante
    W_new = W_frozen + (B @ A) * scaling
    """
    def __init__(self, in_features, out_features, rank=8, alpha=16, dropout=0.1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Matrices LoRA de faible rang
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x, original_output):
        """
        x: entrée originale
        original_output: sortie de la couche linéaire gelée
        """
        # Calcul de l'adaptation LoRA
        lora_output = self.dropout(x) @ self.lora_A @ self.lora_B
        return original_output + lora_output * self.scaling


class LoRAWrapper(nn.Module):
    """
    Wrapper pour appliquer LoRA à un modèle GPT2
    """
    def __init__(self, base_model, rank=8, alpha=16, dropout=0.1, target_modules=None):
        super().__init__()
        self.base_model = base_model
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        
        # Par défaut, on applique LoRA aux projections Q, K, V et FFN
        if target_modules is None:
            target_modules = ['q_proj', 'k_proj', 'v_proj', 'fc1', 'fc2']
        self.target_modules = target_modules
        
        # Geler tous les paramètres du modèle de base
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Ajouter les couches LoRA
        self.lora_layers = nn.ModuleDict()
        self.name_mapping = {}  # Correspondance nom_original -> nom_safe
        self._inject_lora()
        
        print(f"\n🔧 LoRA Configuration:")
        print(f"   - Rank: {rank}")
        print(f"   - Alpha: {alpha}")
        print(f"   - Dropout: {dropout}")
        print(f"   - Target modules: {target_modules}")
        print(f"   - LoRA layers added: {len(self.lora_layers)}")
        print(f"   - Trainable params: {self.count_trainable_params():,}")
        print(f"   - Total params: {self.count_total_params():,}")
        print(f"   - Trainable %: {100 * self.count_trainable_params() / self.count_total_params():.2f}%")
    
    def _inject_lora(self):
        """Injecte les couches LoRA dans le modèle"""
        for name, module in self.base_model.named_modules():
            # Chercher les couches linéaires qui correspondent aux modules cibles
            if isinstance(module, nn.Linear):
                # Vérifier si le nom contient un des modules cibles
                for target in self.target_modules:
                    if target in name:
                        lora_layer = LoRALayer(
                            module.in_features,
                            module.out_features,
                            rank=self.rank,
                            alpha=self.alpha,
                            dropout=self.dropout
                        )
                        # Remplacer les points par des underscores pour ModuleDict
                        safe_name = name.replace('.', '_')
                        self.lora_layers[safe_name] = lora_layer
                        self.name_mapping[name] = safe_name
                        break
    
    def forward(self, input_ids, attention_mask=None):
        """Forward pass avec LoRA"""
        # Hook pour intercepter et modifier les sorties
        handles = []
        
        def make_hook(lora_layer):
            def hook(module, input, output):
                # Appliquer LoRA à la sortie
                return lora_layer(input[0], output)
            return hook
        
        # Créer un dictionnaire des modules une seule fois
        modules_dict = dict(self.base_model.named_modules())
        
        # Attacher les hooks aux modules concernés
        for orig_name, safe_name in self.name_mapping.items():
            lora_layer = self.lora_layers[safe_name]
            module = modules_dict[orig_name]
            handle = module.register_forward_hook(make_hook(lora_layer))
            handles.append(handle)
        
        # Forward pass du modèle de base
        logits, hidden_states = self.base_model(input_ids)
        
        # Retirer les hooks
        for handle in handles:
            handle.remove()
        
        return logits, hidden_states
    
    def count_trainable_params(self):
        """Compte le nombre de paramètres entraînables (LoRA uniquement)"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def count_total_params(self):
        """Compte le nombre total de paramètres"""
        return sum(p.numel() for p in self.parameters())
    
    def save_lora_weights(self, path):
        """Sauvegarde uniquement les poids LoRA"""
        lora_state = {
            'lora_layers': self.lora_layers.state_dict(),
            'config': {
                'rank': self.rank,
                'alpha': self.alpha,
                'dropout': self.dropout,
                'target_modules': self.target_modules
            }
        }
        torch.save(lora_state, path)
        print(f"✅ Poids LoRA sauvegardés: {path}")
    
    def load_lora_weights(self, path):
        """Charge les poids LoRA"""
        lora_state = torch.load(path, map_location=next(self.parameters()).device)
        self.lora_layers.load_state_dict(lora_state['lora_layers'])
        print(f"✅ Poids LoRA chargés: {path}")
    
    def merge_and_save_full_model(self, path):
        """Fusionne LoRA avec le modèle de base et sauvegarde le tout"""
        # Dégeler temporairement les paramètres
        for param in self.base_model.parameters():
            param.requires_grad = True
        
        # Créer un dictionnaire des modules
        modules_dict = dict(self.base_model.named_modules())
        
        # Fusionner LoRA dans les poids du modèle de base
        for orig_name, safe_name in self.name_mapping.items():
            lora_layer = self.lora_layers[safe_name]
            module = modules_dict[orig_name]
            if isinstance(module, nn.Linear):
                # W_new = W_old + (B @ A) * scaling
                delta_w = (lora_layer.lora_A @ lora_layer.lora_B) * lora_layer.scaling
                module.weight.data += delta_w.T
        
        # Sauvegarder le modèle fusionné
        torch.save(self.base_model.state_dict(), path)
        print(f"✅ Modèle fusionné sauvegardé: {path}")
        
        # Regeler les paramètres
        for param in self.base_model.parameters():
            param.requires_grad = False


# ============================================================================
# CLASSES EXISTANTES (WikipediaScraper, OASST1DialogueLoader, etc.)
# ============================================================================

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
            return []
        
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
            print("⚠️ Aucune conversation extraite")
        else:
            print(f"✅ {len(dialogues)} conversations extraites")
        
        return dialogues
    
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
    """Dataset qui applique automatiquement l'instruction tuning"""
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


# ============================================================================
# TRAINER AVEC LoRA
# ============================================================================

class LoRATrainer:
    def __init__(
        self,
        model_dir,
        tokenizer_path,
        device,
        language='fr',
        instruction_template="chat_bot",
        custom_data_dir=None,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=None
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
        
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules
        
        self.wiki_scraper = WikipediaScraper(language)
        self.wiki_qa_gen = WikiQAGenerator()
        self.dialogue_loader = OASST1DialogueLoader(language, batch_size=50)
        
        os.makedirs(model_dir, exist_ok=True)
        
        self.base_model, self.tokenizer, self.config = self._load_base_model()
        self.model = self._wrap_with_lora()
        
        self.history_file = os.path.join(model_dir, "lora_training_history.json")
        self.history = self._load_history()
        
        self.topics_file = os.path.join(model_dir, "trained_topics.json")
        self.topics = self._load_topics()
        
        print(f"\n✅ LoRA + Instruction Tuning activé")
        print(f"   Template: {instruction_template}")
        print(f"   LoRA Rank: {lora_rank}, Alpha: {lora_alpha}")

    def _load_base_model(self):
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
            print(f"✅ Chargement du modèle de base : {model_path}")
            try:
                state = torch.load(model_path, map_location=self.device, weights_only=True)
            except TypeError:
                state = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state)
        else:
            print("🆕 Initialisation d'un nouveau modèle de base")
        
        model.to(self.device)
        return model, tokenizer, cfg
    
    def _wrap_with_lora(self):
        """Enveloppe le modèle de base avec LoRA"""
        lora_model = LoRAWrapper(
            self.base_model,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            dropout=self.lora_dropout,
            target_modules=self.target_modules
        )
        
        # Charger les poids LoRA existants si disponibles
        lora_path = os.path.join(self.model_dir, "lora_weights.pt")
        if os.path.exists(lora_path):
            print(f"✅ Chargement des poids LoRA existants")
            lora_model.load_lora_weights(lora_path)
        
        return lora_model

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
            "instruction_template_used": self.instruction_template,
            "lora_config": {
                "rank": self.lora_rank,
                "alpha": self.lora_alpha,
                "dropout": self.lora_dropout
            }
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
        """Charge les données personnalisées depuis le dossier data/"""
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
        print("🔄 GÉNÉRATION DATASET MIXTE AVEC INSTRUCTION TUNING + LoRA")
        print("="*60)
        
        dataset = []
        wiki_topics_this_cycle = []
        dialogue_samples_this_cycle = []
        categories_count = {}
        
        # SOURCE 1: Données personnalisées
        custom_data = []
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
        print(f"   - Données personnalisées: {len(custom_data) * repeat_important if custom_data else 0}")
        print(f"   - Wikipedia Q&A: {wiki_count}")
        print(f"   - OASST1 dialogues: {dialogue_count * repeat_important}")
        print(f"🎯 Instruction Template: {self.instruction_template}")
        print(f"🔧 LoRA: rank={self.lora_rank}, alpha={self.lora_alpha}")
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
        lr=5e-4,
        use_custom_data=True,
        save_merged=False
    ):
        print("\n" + "="*70)
        print("🚀 DÉMARRAGE CYCLE D'ENTRAÎNEMENT LoRA + INSTRUCTION TUNING")
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
        
        # Optimiser uniquement les paramètres LoRA
        optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=lr
        )
        loss_fn = CrossEntropyLoss(ignore_index=-100)
        
        self.model.train()
        
        total_loss = 0
        step = 0
        
        print(f"\n⏳ Entraînement sur {len(dataset)} exemples, {epochs} époques")
        print(f"📊 Paramètres entraînables: {self.model.count_trainable_params():,}")
        
        for epoch in range(epochs):
            epoch_loss = 0
            pbar = tqdm(dataloader, desc=f"Époque {epoch+1}/{epochs}")
            
            for batch in pbar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                logits, _ = self.model(input_ids, attention_mask)
                
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
        
        # Sauvegarder les poids LoRA
        lora_path = os.path.join(self.model_dir, "lora_weights.pt")
        self.model.save_lora_weights(lora_path)
        
        # Optionnellement fusionner et sauvegarder le modèle complet
        if save_merged:
            merged_path = os.path.join(self.model_dir, "model_merged.pt")
            self.model.merge_and_save_full_model(merged_path)
        
        cycle_info = {
            "cycle": len(self.history["cycles"]) + 1,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "examples": len(dataset),
            "epochs": epochs,
            "avg_loss": avg_loss,
            "instruction_template": self.instruction_template,
            "lora_config": {
                "rank": self.lora_rank,
                "alpha": self.lora_alpha,
                "trainable_params": self.model.count_trainable_params()
            }
        }
        self.history["cycles"].append(cycle_info)
        self.history["total_qa_trained"] += len(dataset)
        self._save_history()
        
        print("\n" + "="*70)
        print(f"✅ CYCLE {cycle_info['cycle']} TERMINÉ")
        print(f"   Loss moyenne: {avg_loss:.4f}")
        print(f"   Paramètres LoRA entraînés: {self.model.count_trainable_params():,}")
        print(f"   Total exemples entraînés (historique): {self.history['total_qa_trained']}")
        print("="*70)

    def display_stats(self):
        print("\n" + "="*60)
        print("📊 STATISTIQUES D'ENTRAÎNEMENT LoRA")
        print("="*60)
        
        print(f"\n🔢 Cycles d'entraînement: {len(self.history['cycles'])}")
        print(f"📝 Total exemples entraînés: {self.history['total_qa_trained']}")
        print(f"🎯 Template d'instruction: {self.instruction_template}")
        
        if 'lora_config' in self.history:
            lora_cfg = self.history['lora_config']
            print(f"\n🔧 Configuration LoRA:")
            print(f"   - Rank: {lora_cfg.get('rank', 'N/A')}")
            print(f"   - Alpha: {lora_cfg.get('alpha', 'N/A')}")
            print(f"   - Dropout: {lora_cfg.get('dropout', 'N/A')}")
        
        print(f"\n💾 Efficacité mémoire:")
        print(f"   - Paramètres entraînables: {self.model.count_trainable_params():,}")
        print(f"   - Paramètres totaux: {self.model.count_total_params():,}")
        print(f"   - Ratio: {100 * self.model.count_trainable_params() / self.model.count_total_params():.2f}%")
        
        if self.history['cycles']:
            last_cycle = self.history['cycles'][-1]
            print(f"\n🕐 Dernier cycle:")
            print(f"   - Date: {last_cycle['timestamp']}")
            print(f"   - Exemples: {last_cycle['examples']}")
            print(f"   - Loss: {last_cycle['avg_loss']:.4f}")
            if 'lora_config' in last_cycle:
                print(f"   - Paramètres LoRA: {last_cycle['lora_config'].get('trainable_params', 'N/A'):,}")
        
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
    
    def export_for_inference(self, output_dir):
        """Exporte le modèle pour l'inférence (modèle de base + poids LoRA)"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Copier le modèle de base
        base_model_path = os.path.join(self.model_dir, "model.pt")
        if os.path.exists(base_model_path):
            import shutil
            shutil.copy(base_model_path, os.path.join(output_dir, "base_model.pt"))
        
        # Copier les poids LoRA
        lora_path = os.path.join(self.model_dir, "lora_weights.pt")
        if os.path.exists(lora_path):
            import shutil
            shutil.copy(lora_path, os.path.join(output_dir, "lora_weights.pt"))
        
        # Copier la config
        config_path = os.path.join(self.model_dir, "config.json")
        if os.path.exists(config_path):
            import shutil
            shutil.copy(config_path, os.path.join(output_dir, "config.json"))
        
        # Sauvegarder les infos LoRA
        lora_info = {
            "rank": self.lora_rank,
            "alpha": self.lora_alpha,
            "dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "instruction_template": self.instruction_template
        }
        with open(os.path.join(output_dir, "lora_config.json"), 'w') as f:
            json.dump(lora_info, f, indent=2)
        
        print(f"\n✅ Modèle exporté vers: {output_dir}")
        print("   Fichiers:")
        print("   - base_model.pt (modèle de base gelé)")
        print("   - lora_weights.pt (adaptations LoRA)")
        print("   - config.json (configuration du modèle)")
        print("   - lora_config.json (configuration LoRA)")


def main():
    print("\n" + "="*70)
    print("🤖 SYSTÈME D'ENTRAÎNEMENT LLM AVEC LoRA + INSTRUCTION TUNING")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Device: {device}")
    
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
    
    print(f"📁 Model directory: {model_dir}")
    print(f"🔤 Tokenizer: {tokenizer_path}")
    
    # Configuration LoRA
    print("\n🔧 Configuration LoRA:")
    print("   - Rank: 8 (faible rang pour efficacité)")
    print("   - Alpha: 16 (facteur de scaling)")
    print("   - Dropout: 0.1 (régularisation)")
    print("   - Modules ciblés: Q, K, V projections + FFN")
    
    trainer = LoRATrainer(
        model_dir=model_dir,
        tokenizer_path=tokenizer_path,
        device=device,
        language='en',
        instruction_template="chat_bot",
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'fc1', 'fc2']
    )
    
    trainer.display_stats()
    
    print("\n🎯 Démarrage de l'entraînement LoRA...")
    print("   - 10 articles Wikipedia (English)")
    print("   - 100 dialogues OASST1 (English)")
    print("   - 3 époques")
    print("   - Batch size: 4")
    print("   - Learning rate: 5e-4 (plus élevé pour LoRA)")
    print("   - Template: chat_bot (Human/Bot)")


    for cycle in range(100):  # 100 cycles
        print(f"\n{'='*70}")
        print(f"🔄 CYCLE {cycle + 1}/100")
        print(f"{'='*70}")
        trainer.train_one_cycle(
            num_articles=20,
            qa_per_article=5,
            num_dialogues=200,
            epochs=3,
            batch_size=4,
            lr=5e-4,
            use_custom_data=False,
            save_merged=False
        )
       # Sauvegarder périodiquement le modèle fusionné
    if (cycle + 1) % 10 == 0:
        trainer.model.merge_and_save_full_model(
            os.path.join(trainer.model_dir, f"model_checkpoint_cycle_{cycle+1}.pt")
        )
    trainer.display_stats()
    
    # Exporter pour l'inférence
    export_dir = os.path.join(model_dir, "export")
    trainer.export_for_inference(export_dir)
    
    print("\n✅ Entraînement LoRA terminé!")
    print("\n💡 Avantages de LoRA:")
    print("   ✓ Entraîne seulement ~1% des paramètres")
    print("   ✓ Beaucoup moins de mémoire requise")
    print("   ✓ Entraînement plus rapide")
    print("   ✓ Modèle de base reste intact")
    print("   ✓ Plusieurs adaptations LoRA possibles sur le même modèle")
    print("\n📊 Sources: Wikipedia + OASST1 (Hugging Face)")
    print("🎯 Format: Instruction Tuning automatique")


if __name__ == "__main__":
    main()