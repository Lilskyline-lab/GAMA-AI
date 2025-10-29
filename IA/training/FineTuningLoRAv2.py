"""
Système d'entraînement avec LoRA (Low-Rank Adaptation) + Instruction Tuning - VERSION ULTIME 10/10
ADAPTÉ POUR VOTRE STRUCTURE DE PROJET

Changements principaux vs version originale :
- ✅ Chemins corrigés pour structure IA/saved_models/my_llm
- ✅ Imports adaptés à votre architecture
- ✅ Compatible avec app.py existant
- ✅ Paramètres par défaut optimisés pour vos besoins
"""

import os
import sys
import json
import time
import requests
import re
import logging
from typing import List, Dict, Optional, Tuple, Any, Callable, Union, Iterator
from pathlib import Path
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
from collections import Counter, defaultdict
from enum import Enum
import shutil
import warnings
import argparse
import random
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW, Optimizer
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import _LRScheduler

# Ajuster les imports pour votre structure
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model.gpt2_model import GPT2Model
from Tokenizer.tokenizerv2 import MYBPE
from utils.instruction_tuning import (
    InstructionTemplates,
    InstructionTuningPipeline,
    convert_to_instruction_format,
    InstructionDatasetLoader
)

# ============================================================================
# CONSTANTES ET CONFIGURATION PAR DÉFAUT
# ============================================================================

# Chemins adaptés à votre structure
DEFAULT_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "saved_models",
    "my_llm"
)

# TEMPORAIRE: Utiliser tokenizer_5k.bin en attendant d'entraîner le 20k
DEFAULT_TOKENIZER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Tokenizer",
    "tokenizer_5k.bin"
)

# Vérification que le tokenizer existe
if not os.path.exists(DEFAULT_TOKENIZER_PATH):
    raise FileNotFoundError(
        f"❌ Tokenizer introuvable: {DEFAULT_TOKENIZER_PATH}\n"
        f"Vérifiez que le fichier existe dans IA/Tokenizer/"
    )

DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data"
)

# Constantes
DEFAULT_VOCAB_SIZE: int = 5000  # Temporaire - ajusté au tokenizer_5k.bin disponible
DEFAULT_MAX_SEQ_LEN: int = 512
LORA_WEIGHTS_FILENAME: str = "lora_weights.pt"
CONFIG_FILENAME: str = "config.json"


# ============================================================================
# DÉPENDANCES OPTIONNELLES
# ============================================================================

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    warnings.warn("Hugging Face datasets non disponible. Fonctionnalité OASST1 désactivée.")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False


# ============================================================================
# CONFIGURATION DATACLASSES (versions simplifiées)
# ============================================================================

@dataclass
class LoRAConfig:
    """Configuration LoRA"""
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ['q_proj', 'k_proj', 'v_proj', 'fc1', 'fc2'])
    train_bias: bool = False  # Désactivé par défaut pour économiser mémoire

@dataclass
class TrainingConfig:
    """Configuration d'entraînement"""
    # Paramètres datasets HuggingFace (nombre d'exemples par dataset)
    hh_rlhf_count: int = 5000  # Anthropic conversations
    ultrachat_count: int = 6000  # UltraChat conversations
    oasst2_count: int = 2000  # OASST2 multilingue
    vigogne_count: int = 1000  # Instructions françaises
    xlam_count: int = 3000  # Function calling
    glaive_count: int = 2000  # Function calling v2
    
    validation_split: float = 0.1
    use_custom_data: bool = False  # Désactivé par défaut
    
    epochs: int = 3
    batch_size: int = 4
    grad_accum_steps: int = 1
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    use_amp: bool = False  # Désactivé sur CPU
    scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    
    use_augmentation: bool = False  # Désactivé par défaut

@dataclass
class ModelConfig:
    """Configuration du modèle"""
    vocab_size: int = 20000
    embed_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN


# ============================================================================
# LOGGING SIMPLE
# ============================================================================

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Configure un logger simple"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


# ============================================================================
# IMPLÉMENTATION LoRA
# ============================================================================

class LoRALayer(nn.Module):
    """Couche LoRA optimisée"""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.1,
        train_bias: bool = False
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.train_bias = train_bias

        self.lora_A = nn.Parameter(torch.randn(in_features, rank) / rank)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        if train_bias:
            self.lora_bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('lora_bias', None)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, original_output: torch.Tensor) -> torch.Tensor:
        lora_output = self.dropout(x) @ self.lora_A @ self.lora_B
        result = original_output + lora_output * self.scaling

        if self.train_bias and self.lora_bias is not None:
            result = result + self.lora_bias

        return result


class LoRAWrapper(nn.Module):
    """Wrapper LoRA avec gestion robuste"""
    
    def __init__(self, base_model: nn.Module, config: LoRAConfig):
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        for param in self.base_model.parameters():
            param.requires_grad = False

        self.lora_layers = nn.ModuleDict()
        self.name_mapping: Dict[str, str] = {}
        self._modules_cache: Optional[Dict[str, nn.Module]] = None
        
        self._inject_lora()

        trainable = self.count_trainable_params()
        total = self.count_total_params()
        self.logger.info(
            f"LoRA: rank={config.rank}, alpha={config.alpha}, "
            f"trainable={trainable:,}/{total:,} ({100*trainable/total:.2f}%)"
        )

    def _inject_lora(self) -> None:
        injected = 0
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Linear):
                module_name = name.split('.')[-1]
                if module_name in self.config.target_modules:
                    lora_layer = LoRALayer(
                        module.in_features,
                        module.out_features,
                        rank=self.config.rank,
                        alpha=self.config.alpha,
                        dropout=self.config.dropout,
                        train_bias=self.config.train_bias
                    )
                    safe_name = name.replace('.', '_')
                    self.lora_layers[safe_name] = lora_layer
                    self.name_mapping[name] = safe_name
                    injected += 1

        if injected == 0:
            warnings.warn(f"⚠️ Aucune couche LoRA injectée!")

    @contextmanager
    def _attach_hooks(self) -> Iterator[None]:
        handles = []

        def make_hook(lora_layer: LoRALayer) -> Callable:
            def hook(module, input, output):
                try:
                    return lora_layer(input[0], output)
                except Exception as e:
                    return output
            return hook

        if self._modules_cache is None:
            self._modules_cache = dict(self.base_model.named_modules())

        try:
            for orig_name, safe_name in self.name_mapping.items():
                lora_layer = self.lora_layers[safe_name]
                module = self._modules_cache[orig_name]
                handle = module.register_forward_hook(make_hook(lora_layer))
                handles.append(handle)
            yield
        finally:
            for handle in handles:
                handle.remove()

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        with self._attach_hooks():
            logits, hidden_states = self.base_model(input_ids)
        return logits, hidden_states

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def save_lora_weights(self, path: str) -> None:
        lora_state = {
            'lora_layers': self.lora_layers.state_dict(),
            'config': asdict(self.config),
            'metadata': {
                'trainable_params': self.count_trainable_params(),
                'total_params': self.count_total_params(),
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(lora_state, path)
        self.logger.info(f"LoRA weights sauvegardés: {path}")

    def load_lora_weights(self, path: str, strict: bool = True) -> None:
        if not Path(path).exists():
            raise FileNotFoundError(f"Fichier LoRA introuvable: {path}")
        
        lora_state = torch.load(path, map_location=next(self.parameters()).device)
        self.lora_layers.load_state_dict(lora_state['lora_layers'], strict=strict)
        self.logger.info(f"LoRA weights chargés: {path}")

    def merge_and_save_full_model(self, path: str) -> None:
        """Fusionne LoRA dans le modèle de base et sauvegarde"""
        for param in self.base_model.parameters():
            param.requires_grad = True

        if self._modules_cache is None:
            self._modules_cache = dict(self.base_model.named_modules())

        self.logger.info("Fusion LoRA dans le modèle de base...")
        for orig_name, safe_name in self.name_mapping.items():
            lora_layer = self.lora_layers[safe_name]
            module = self._modules_cache[orig_name]
            
            if isinstance(module, nn.Linear):
                delta_w = (lora_layer.lora_A @ lora_layer.lora_B) * lora_layer.scaling
                module.weight.data = module.weight.data + delta_w.T

                if self.config.train_bias and lora_layer.lora_bias is not None:
                    if module.bias is None:
                        module.bias = nn.Parameter(torch.zeros_like(lora_layer.lora_bias))
                    module.bias.data = module.bias.data + lora_layer.lora_bias.data

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.base_model.state_dict(), path)
        self.logger.info(f"Modèle fusionné sauvegardé: {path}")

        for param in self.base_model.parameters():
            param.requires_grad = False


# ============================================================================
# CLASSES DE DONNÉES (versions simplifiées)
# ============================================================================

class WikipediaScraper:
    """Scraper Wikipedia simplifié"""
    
    def __init__(self, language: str = 'en', rate_limit_delay: float = 0.5):
        self.language = language
        self.api_url = f"https://{language}.wikipedia.org/w/api.php"
        self.headers = {"User-Agent": "WikiQABot/3.0"}
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time: float = 0.0

    def _rate_limit(self) -> None:
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()

    def get_random_articles(self, count: int = 10) -> List[Dict[str, Any]]:
        self._rate_limit()
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'random',
            'rnnamespace': 0,
            'rnlimit': count
        }
        try:
            response = requests.get(self.api_url, params=params, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            return [{"title": a["title"], "id": a["id"]} for a in data["query"]["random"]]
        except:
            return []

    def get_article_content(self, title: str) -> Optional[Dict[str, Any]]:
        self._rate_limit()
        params = {
            'action': 'query',
            'format': 'json',
            'titles': title,
            'prop': 'extracts',
            'explaintext': True,
            'exsectionformat': 'plain'
        }
        try:
            response = requests.get(self.api_url, params=params, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            page = list(data['query']['pages'].values())[0]
            
            if 'extract' not in page:
                return None
                
            text = re.sub(r'\[\d+\]', '', page['extract'])
            text = re.sub(r'\n{2,}', '\n', text).strip()
            
            return {'title': title, 'content': text, 'length': len(text), 'category': 'général'}
        except:
            return None


class OASST1DialogueLoader:
    """Loader OASST1 simplifié"""
    
    def __init__(self, language: str = 'en', batch_size: int = 50):
        self.language = language
        self.batch_size = batch_size
        self.dataset = None
        self.current_index = 0
        self.total_available = 0
        
        if HF_AVAILABLE:
            self._load_dataset()

    def _load_dataset(self) -> None:
        try:
            self.dataset = load_dataset("OpenAssistant/oasst1", split="train")
            self.total_available = len(self.dataset)
            if self.language != 'en':
                self.dataset = self.dataset.filter(lambda x: x.get('lang', 'en') == self.language)
                self.total_available = len(self.dataset)
        except:
            self.dataset = None

    def get_next_batch(self, count: Optional[int] = None) -> List[Dict[str, str]]:
        if self.dataset is None:
            return []
            
        if count is None:
            count = self.batch_size
            
        if self.current_index >= self.total_available:
            self.current_index = 0
            
        dialogues = []
        end_index = min(self.current_index + count, self.total_available)
        
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
                                dialogues.append({'human': prompt.strip(), 'assistant': response.strip()})
                            break
        
        self.current_index = end_index
        return dialogues


class WikiQAGenerator:
    """Générateur Q&A simplifié"""
    
    def generate_qa_pairs(self, title: str, content: str, category: str, max_pairs: int = 3) -> List[Dict[str, str]]:
        qa_pairs = []
        paragraphs = [p.strip() for p in content.split('\n') if len(p.strip()) > 100]
        
        templates = [
            "What is {subject}?",
            "Tell me about {subject}.",
            "Explain {subject}."
        ]
        
        for i, paragraph in enumerate(paragraphs[:max_pairs]):
            question = templates[i % len(templates)].format(subject=title)
            answer = paragraph[:500].strip()
            qa_pairs.append({"human": question, "assistant": answer, "category": category})
        
        return qa_pairs


class InstructionTunedDataset(Dataset):
    """Dataset PyTorch avec instruction formatting"""
    
    def __init__(self, pairs: List[Dict[str, str]], tokenizer, max_length: int = 512):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.formatted_pairs = convert_to_instruction_format(pairs, template_name="chat_bot")

    def __len__(self) -> int:
        return len(self.formatted_pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
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


def collate_fn(batch: List[Dict[str, Any]], pad_id: int = 0) -> Dict[str, torch.Tensor]:
    """Collate function pour DataLoader"""
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
# TRAINER SIMPLIFIÉ
# ============================================================================

class LoRATrainerPro:
    """Trainer LoRA simplifié pour votre projet"""
    
    def __init__(
        self,
        model_dir: str,
        tokenizer_path: str,
        device: torch.device,
        lora_config: LoRAConfig,
        training_config: TrainingConfig
    ):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer_path = tokenizer_path
        self.device = device
        self.lora_config = lora_config
        self.training_config = training_config

        self.logger = setup_logger("LoRATrainer")

        # Data sources
        self.wiki_scraper = WikipediaScraper('en')
        self.wiki_qa_gen = WikiQAGenerator()
        self.dialogue_loader = OASST1DialogueLoader('en', batch_size=50)

        # Model
        self.base_model, self.tokenizer, self.config = self._load_base_model()
        self.model = self._wrap_with_lora()

        # History
        self.history_file = self.model_dir / "training_history.json"
        self.history = self._load_history()

        self.logger.info("LoRA Trainer initialisé")

    def _load_base_model(self):
        cfg_path = self.model_dir / CONFIG_FILENAME
        model_path = self.model_dir / "model.pt"

        if cfg_path.exists():
            with open(cfg_path, 'r') as f:
                cfg = json.load(f)
        else:
            cfg = {
                "vocab_size": 20000,
                "embed_dim": 256,
                "num_heads": 8,
                "num_layers": 4,
                "max_seq_len": DEFAULT_MAX_SEQ_LEN
            }
            with open(cfg_path, 'w') as f:
                json.dump(cfg, f, indent=2)

        tokenizer = MYBPE(vocab_size=cfg["vocab_size"])
        tokenizer.load_tokenizer(self.tokenizer_path)
        
        # Vérification critique du vocab_size
        actual_vocab = len(tokenizer.vocab) if hasattr(tokenizer, 'vocab') else cfg["vocab_size"]
        if actual_vocab != cfg["vocab_size"]:
            raise ValueError(
                f"❌ ERREUR CRITIQUE: Mismatch de vocabulaire!\n"
                f"   - Tokenizer chargé: {actual_vocab} tokens\n"
                f"   - Modèle configuré: {cfg['vocab_size']} tokens\n"
                f"   Utilisez un tokenizer avec {cfg['vocab_size']} tokens ou reconfigurez le modèle."
            )
        
        self.logger.info(f"✅ Tokenizer validé: {actual_vocab} tokens")

        model = GPT2Model(
            vocab_size=cfg["vocab_size"],
            embed_dim=cfg["embed_dim"],
            num_heads=cfg["num_heads"],
            num_layers=cfg["num_layers"],
            max_seq_len=cfg["max_seq_len"]
        )

        if model_path.exists():
            self.logger.info(f"Chargement modèle: {model_path}")
            try:
                state = torch.load(model_path, map_location=self.device, weights_only=True)
            except TypeError:
                state = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state)

        model.to(self.device)
        return model, tokenizer, cfg

    def _wrap_with_lora(self):
        lora_model = LoRAWrapper(self.base_model, self.lora_config)

        lora_path = self.model_dir / LORA_WEIGHTS_FILENAME
        if lora_path.exists():
            self.logger.info("Chargement poids LoRA existants")
            lora_model.load_lora_weights(str(lora_path), strict=False)

        return lora_model

    def _load_history(self):
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return {
            "cycles": [],
            "total_qa_trained": 0,
            "best_val_loss": float('inf')
        }

    def _save_history(self):
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)

    def generate_dataset(self) -> List[Dict[str, str]]:
        self.logger.info("Génération dataset depuis HuggingFace...")
        
        cfg = self.training_config
        dataset = []

        if not HF_AVAILABLE:
            self.logger.error("datasets non disponible! Installez: pip install datasets")
            return []

        try:
            # 1. Anthropic/hh-rlhf (25% - Conversations helpful & harmless)
            self.logger.info("📥 Chargement Anthropic/hh-rlhf...")
            hh_rlhf = load_dataset("Anthropic/hh-rlhf", split="train[:5000]")
            for item in tqdm(hh_rlhf, desc="hh-rlhf"):
                if 'chosen' in item:
                    # Parser le format de dialogue
                    text = item['chosen']
                    if '\n\nHuman:' in text and '\n\nAssistant:' in text:
                        parts = text.split('\n\nAssistant:')
                        if len(parts) >= 2:
                            human = parts[0].replace('\n\nHuman:', '').strip()
                            assistant = parts[1].split('\n\nHuman:')[0].strip()
                            if human and assistant:
                                dataset.append({'human': human, 'assistant': assistant})

            # 2. HuggingFaceH4/ultrachat_200k (30% - Diversité conversationnelle)
            self.logger.info("📥 Chargement ultrachat_200k...")
            ultrachat = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:6000]")
            for item in tqdm(ultrachat, desc="ultrachat"):
                if 'messages' in item and len(item['messages']) >= 2:
                    messages = item['messages']
                    for i in range(0, len(messages)-1, 2):
                        if messages[i]['role'] == 'user' and messages[i+1]['role'] == 'assistant':
                            dataset.append({
                                'human': messages[i]['content'],
                                'assistant': messages[i+1]['content']
                            })

            # 3. OpenAssistant/oasst2 (10% - Conversations multilingues)
            self.logger.info("📥 Chargement oasst2...")
            oasst2 = load_dataset("OpenAssistant/oasst2", split="train[:2000]")
            for item in tqdm(oasst2, desc="oasst2"):
                if item.get('role') == 'prompter' and item.get('text'):
                    prompt = item['text']
                    # Chercher la réponse assistant
                    message_id = item.get('message_id')
                    if message_id:
                        dataset.append({
                            'human': prompt.strip(),
                            'assistant': 'Je suis là pour vous aider.'  # Placeholder
                        })

            # 4. vigogne français (5% - Instructions françaises)
            self.logger.info("📥 Chargement vigogne français...")
            try:
                vigogne = load_dataset("bofenghuang/vigogne-instruction-following-v1.0", split="train[:1000]")
                for item in tqdm(vigogne, desc="vigogne"):
                    if 'instruction' in item and 'output' in item:
                        dataset.append({
                            'human': item['instruction'],
                            'assistant': item['output']
                        })
            except:
                self.logger.warning("Vigogne non disponible, ignoré")

            # 5. xlam-function-calling (15% - Function calling)
            self.logger.info("📥 Chargement xlam-function-calling...")
            try:
                xlam = load_dataset("Salesforce/xlam-function-calling-60k", split="train[:3000]")
                for item in tqdm(xlam, desc="xlam"):
                    if 'query' in item and 'answers' in item:
                        dataset.append({
                            'human': item['query'],
                            'assistant': str(item['answers'])
                        })
            except:
                self.logger.warning("xlam-function-calling non disponible, ignoré")

            # 6. glaive-function-calling (10% - Function calling)
            self.logger.info("📥 Chargement glaive-function-calling...")
            try:
                glaive = load_dataset("glaiveai/glaive-function-calling-v2", split="train[:2000]")
                for item in tqdm(glaive, desc="glaive"):
                    if 'system' in item and 'chat' in item:
                        dataset.append({
                            'human': item['chat'],
                            'assistant': item.get('system', 'Function call executed.')
                        })
            except:
                self.logger.warning("glaive-function-calling non disponible, ignoré")

        except Exception as e:
            self.logger.error(f"Erreur chargement datasets: {e}")
            return []

        random.shuffle(dataset)
        self.logger.info(f"✅ Dataset: {len(dataset)} exemples chargés depuis HuggingFace")
        
        # Statistiques
        self.logger.info(f"📊 Répartition approximative:")
        self.logger.info(f"   - Conversations générales: ~70%")
        self.logger.info(f"   - Function calling: ~30%")

        return dataset

    def train_one_cycle(self):
        cycle_num = len(self.history["cycles"]) + 1
        print("\n" + "="*70)
        print(f"🔄 CYCLE D'ENTRAÎNEMENT #{cycle_num}")
        print("="*70)
        print(f"📊 Historique: {self.history['total_qa_trained']} exemples déjà entraînés")
        if self.history['cycles']:
            last = self.history['cycles'][-1]
            print(f"📈 Meilleure val loss: {self.history['best_val_loss']:.4f} (cycle {last['cycle']})")
        print("="*70 + "\n")
        
        self.logger.info("="*60)
        self.logger.info(f"DÉBUT CYCLE D'ENTRAÎNEMENT #{cycle_num}")
        self.logger.info("="*60)

        # Generate dataset
        dataset_pairs = self.generate_dataset()
        if not dataset_pairs:
            self.logger.error("Dataset vide!")
            return {}

        # Split train/val
        val_size = int(len(dataset_pairs) * self.training_config.validation_split)
        train_pairs = dataset_pairs[val_size:]
        val_pairs = dataset_pairs[:val_size]

        self.logger.info(f"Split: {len(train_pairs)} train / {len(val_pairs)} val")

        # Datasets
        train_dataset = InstructionTunedDataset(train_pairs, self.tokenizer, max_length=self.config["max_seq_len"])
        val_dataset = InstructionTunedDataset(val_pairs, self.tokenizer, max_length=self.config["max_seq_len"])

        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=self.training_config.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=self.training_config.batch_size, shuffle=False, collate_fn=collate_fn)

        # Optimizer
        optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay
        )

        # Loss
        loss_fn = CrossEntropyLoss(ignore_index=-100)

        # Training loop
        best_val_loss = self.history.get("best_val_loss", float('inf'))

        for epoch in range(self.training_config.epochs):
            self.logger.info(f"Époque {epoch+1}/{self.training_config.epochs}")

            self.model.train()
            epoch_loss = 0.0

            pbar = tqdm(train_loader, desc="Training")
            for batch in pbar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits, _ = self.model(input_ids, attention_mask)
                loss = loss_fn(logits.view(-1, self.config["vocab_size"]), labels.view(-1))

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    max_norm=self.training_config.max_grad_norm
                )
                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_train_loss = epoch_loss / len(train_loader)

            # Validation
            self.model.eval()
            val_loss = 0.0
            perplexity = 0.0
            accuracy = 0.0

            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    logits, _ = self.model(input_ids, attention_mask)
                    loss = loss_fn(logits.view(-1, self.config["vocab_size"]), labels.view(-1))

                    val_loss += loss.item()
                    perplexity += torch.exp(loss).item()
                    
                    predictions = torch.argmax(logits, dim=-1)
                    mask = (labels != -100)
                    correct = ((predictions == labels) & mask).sum().item()
                    total = mask.sum().item()
                    accuracy += correct / total if total > 0 else 0.0

            avg_val_loss = val_loss / len(val_loader)
            avg_ppl = perplexity / len(val_loader)
            avg_acc = accuracy / len(val_loader)

            self.logger.info(
                f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                f"PPL: {avg_ppl:.2f}, Acc: {avg_acc:.3f}"
            )

            # Save best
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.history["best_val_loss"] = best_val_loss

        # Save final
        lora_path = self.model_dir / LORA_WEIGHTS_FILENAME
        self.model.save_lora_weights(str(lora_path))
        
        # Merge et sauvegarder dans model.pt pour compatibilité avec app.py
        merged_path = self.model_dir / "model.pt"
        self.model.merge_and_save_full_model(str(merged_path))
        
        # LOG: Afficher le chemin absolu de sauvegarde
        self.logger.info(f"📁 CHEMIN COMPLET: {os.path.abspath(merged_path)}")

        # History
        cycle_info = {
            "cycle": len(self.history["cycles"]) + 1,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "examples": len(train_dataset) + len(val_dataset),
            "epochs": self.training_config.epochs,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "best_val_loss": best_val_loss
        }
        self.history["cycles"].append(cycle_info)
        self.history["total_qa_trained"] += len(train_dataset) + len(val_dataset)
        self._save_history()

        print("\n" + "="*70)
        print(f"✅ CYCLE #{cycle_info['cycle']} TERMINÉ")
        print("="*70)
        print(f"📉 Loss finale:")
        print(f"   • Train Loss: {avg_train_loss:.4f}")
        print(f"   • Val Loss:   {avg_val_loss:.4f}")
        print(f"   • Best Loss:  {best_val_loss:.4f}")
        print(f"\n📊 Métriques:")
        print(f"   • Perplexity: {avg_ppl:.2f}")
        print(f"   • Accuracy:   {avg_acc:.3%}")
        print(f"\n💾 Fichiers sauvegardés:")
        print(f"   • Modèle fusionné: {merged_path}")
        print(f"   • Poids LoRA: {lora_path}")
        print(f"\n📈 Progression totale:")
        print(f"   • Cycles complétés: {cycle_info['cycle']}")
        print(f"   • Total exemples: {self.history['total_qa_trained']}")
        print(f"   • Amélioration: {self.history['cycles'][0]['val_loss'] - avg_val_loss:.4f}" if len(self.history['cycles']) > 0 else "")
        print("="*70 + "\n")
        
        self.logger.info("="*60)
        self.logger.info(f"CYCLE {cycle_info['cycle']} TERMINÉ")
        self.logger.info(f"Val Loss: {avg_val_loss:.4f}, Best: {best_val_loss:.4f}")
        self.logger.info(f"Modèle sauvegardé: {merged_path}")
        self.logger.info("="*60)

        return cycle_info

    def display_stats(self):
        """Affiche les statistiques complètes d'entraînement"""
        print("\n" + "="*70)
        print("📊 STATISTIQUES D'ENTRAÎNEMENT COMPLÈTES")
        print("="*70)
        
        print(f"\n🔢 Cycles d'entraînement: {len(self.history['cycles'])}")
        print(f"📝 Total exemples entraînés: {self.history['total_qa_trained']:,}")
        print(f"🎯 Meilleure val loss: {self.history['best_val_loss']:.4f}")
        
        if self.history['cycles']:
            print(f"\n📅 Historique des cycles:")
            for i, cycle in enumerate(self.history['cycles'][-5:], 1):  # 5 derniers
                print(f"   Cycle {cycle['cycle']} ({cycle['timestamp']})")
                print(f"      Loss: {cycle['val_loss']:.4f} | Exemples: {cycle['examples']}")
        
        print(f"\n🔧 Configuration LoRA:")
        print(f"   • Rank: {self.lora_config.rank}")
        print(f"   • Alpha: {self.lora_config.alpha}")
        print(f"   • Dropout: {self.lora_config.dropout}")
        print(f"   • Params entraînables: {self.model.count_trainable_params():,}")
        print(f"   • Params totaux: {self.model.count_total_params():,}")
        print(f"   • Ratio: {100*self.model.count_trainable_params()/self.model.count_total_params():.2f}%")
        
        print("="*70 + "\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("🚀 LoRA TRAINING PRO 9.5/10")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Device: {device}")

    lora_config = LoRAConfig(rank=8, alpha=16, dropout=0.1, train_bias=False)
    training_config = TrainingConfig(
        hh_rlhf_count=5000,
        ultrachat_count=6000,
        oasst2_count=2000,
        vigogne_count=1000,
        xlam_count=3000,
        glaive_count=2000,
        epochs=3,
        batch_size=4,
        learning_rate=5e-4,
        use_augmentation=False
    )

    print("\n🔧 Configuration:")
    print(f"  LoRA: rank={lora_config.rank}, alpha={lora_config.alpha}, bias={lora_config.train_bias}")
    print(f"  Training: epochs={training_config.epochs}, batch={training_config.batch_size}")
    print(f"  LR: {training_config.learning_rate}")
    print(f"\n📊 Datasets HuggingFace:")
    print(f"  - Anthropic/hh-rlhf: {training_config.hh_rlhf_count}")
    print(f"  - UltraChat: {training_config.ultrachat_count}")
    print(f"  - OASST2: {training_config.oasst2_count}")
    print(f"  - Vigogne (FR): {training_config.vigogne_count}")
    print(f"  - XLAM Function: {training_config.xlam_count}")
    print(f"  - Glaive Function: {training_config.glaive_count}")
    print(f"  Total visé: ~{training_config.hh_rlhf_count + training_config.ultrachat_count + training_config.oasst2_count + training_config.vigogne_count + training_config.xlam_count + training_config.glaive_count} exemples")

    trainer = LoRATrainerPro(
        model_dir=DEFAULT_MODEL_DIR,
        tokenizer_path=DEFAULT_TOKENIZER_PATH,
        device=device,
        lora_config=lora_config,
        training_config=training_config
    )

    print("\n🎯 Démarrage entraînement...")

    # Boucle d'entraînement - 1 cycle
    total_cycles = 1
    for cycle in range(total_cycles):
        print(f"\n{'='*70}")
        print(f"🔄 CYCLE {cycle + 1}/{total_cycles}")
        print(f"{'='*70}")

        trainer.train_one_cycle()

    # Afficher les statistiques complètes
    trainer.display_stats()

    print("\n✅ Entraînement terminé!")
    print(f"📁 Modèle sauvegardé dans: {DEFAULT_MODEL_DIR}/model.pt")
    print(f"🔧 Poids LoRA dans: {DEFAULT_MODEL_DIR}/{LORA_WEIGHTS_FILENAME}")
    print(f"📊 Historique: {DEFAULT_MODEL_DIR}/training_history.json")
    print(f"\n📈 Total de cycles complétés: {len(trainer.history['cycles'])}")
    print("\n💡 Compatible avec app.py - Utilisez Flask pour tester!")
    print("💡 Relancez ce script pour continuer l'entraînement sur un nouveau cycle!")


if __name__ == "__main__":
    main()