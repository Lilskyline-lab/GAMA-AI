"""
Module d'Instruction Tuning pour LLM
Formate les données d'entraînement selon différents templates instruction-réponse
Supporte JSON, JSONL, CSV et intégration directe avec le pipeline d'entraînement
"""

import json
import csv
import os
from typing import List, Dict, Union, Optional, Callable
from dataclasses import dataclass
from enum import Enum


class InstructionFormat(Enum):
    """Formats d'instruction disponibles"""
    ALPACA = "alpaca"  # Format Alpaca: instruction + input + output
    CHAT = "chat"  # Format chat: human + assistant
    QA = "qa"  # Format Q&A simple: question + answer
    COMPLETION = "completion"  # Format completion: prompt + completion
    CUSTOM = "custom"  # Format personnalisé


@dataclass
class InstructionTemplate:
    """Template pour formater une instruction"""
    name: str
    format_type: InstructionFormat
    template: str
    fields: List[str]
    
    def format(self, data: Dict) -> str:
        """Formate les données selon le template"""
        try:
            return self.template.format(**data)
        except KeyError as e:
            raise ValueError(f"Champ manquant dans les données: {e}")


class InstructionTemplates:
    """Collection de templates prédéfinis"""
    
    @staticmethod
    def get_alpaca_template() -> InstructionTemplate:
        """Format Alpaca standard"""
        return InstructionTemplate(
            name="alpaca",
            format_type=InstructionFormat.ALPACA,
            template=(
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n"
                "### Input:\n{input}\n\n"
                "### Response:\n{output}"
            ),
            fields=["instruction", "input", "output"]
        )
    
    @staticmethod
    def get_alpaca_no_input_template() -> InstructionTemplate:
        """Format Alpaca sans input"""
        return InstructionTemplate(
            name="alpaca_no_input",
            format_type=InstructionFormat.ALPACA,
            template=(
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n"
                "### Response:\n{output}"
            ),
            fields=["instruction", "output"]
        )
    
    @staticmethod
    def get_chat_template() -> InstructionTemplate:
        """Format Chat (Human/Assistant)"""
        return InstructionTemplate(
            name="chat",
            format_type=InstructionFormat.CHAT,
            template="Human: {human}\nAssistant: {assistant}",
            fields=["human", "assistant"]
        )
    
    @staticmethod
    def get_chat_bot_template() -> InstructionTemplate:
        """Format Chat (Human/Bot) - compatible avec le code existant"""
        return InstructionTemplate(
            name="chat_bot",
            format_type=InstructionFormat.CHAT,
            template="Human: {human}\nBot: {assistant}",
            fields=["human", "assistant"]
        )
    
    @staticmethod
    def get_qa_template() -> InstructionTemplate:
        """Format Q&A simple"""
        return InstructionTemplate(
            name="qa",
            format_type=InstructionFormat.QA,
            template="Question: {question}\nAnswer: {answer}",
            fields=["question", "answer"]
        )
    
    @staticmethod
    def get_completion_template() -> InstructionTemplate:
        """Format completion simple"""
        return InstructionTemplate(
            name="completion",
            format_type=InstructionFormat.COMPLETION,
            template="{prompt}{completion}",
            fields=["prompt", "completion"]
        )
    
    @staticmethod
    def get_vicuna_template() -> InstructionTemplate:
        """Format Vicuna"""
        return InstructionTemplate(
            name="vicuna",
            format_type=InstructionFormat.CHAT,
            template="USER: {user}\nASSISTANT: {assistant}",
            fields=["user", "assistant"]
        )
    
    @staticmethod
    def get_llama2_template() -> InstructionTemplate:
        """Format Llama 2 Chat"""
        return InstructionTemplate(
            name="llama2",
            format_type=InstructionFormat.CHAT,
            template="<s>[INST] {instruction} [/INST] {response} </s>",
            fields=["instruction", "response"]
        )


class InstructionDataFormatter:
    """
    Classe principale pour formater les données d'instruction tuning
    """
    
    def __init__(
        self,
        template: Optional[InstructionTemplate] = None,
        custom_template: Optional[str] = None,
        custom_fields: Optional[List[str]] = None
    ):
        """
        Args:
            template: Template prédéfini (ou None pour chat par défaut)
            custom_template: String de template personnalisé (ex: "Q: {q}\nA: {a}")
            custom_fields: Champs requis pour le template personnalisé
        """
        if custom_template and custom_fields:
            self.template = InstructionTemplate(
                name="custom",
                format_type=InstructionFormat.CUSTOM,
                template=custom_template,
                fields=custom_fields
            )
        elif template:
            self.template = template
        else:
            # Par défaut: format chat compatible avec le code existant
            self.template = InstructionTemplates.get_chat_bot_template()
    
    def format_single(self, data: Dict) -> str:
        """
        Formate une seule paire instruction-réponse
        
        Args:
            data: Dictionnaire contenant les champs nécessaires
            
        Returns:
            String formatée selon le template
        """
        return self.template.format(data)
    
    def format_batch(self, data_list: List[Dict]) -> List[str]:
        """
        Formate un batch de données
        
        Args:
            data_list: Liste de dictionnaires
            
        Returns:
            Liste de strings formatées
        """
        return [self.format_single(data) for data in data_list]
    
    def map_fields(self, data: Dict, field_mapping: Dict[str, str]) -> Dict:
        """
        Mappe les champs d'un dataset vers les champs requis par le template
        
        Args:
            data: Données originales
            field_mapping: Mapping {champ_template: champ_original}
                          Ex: {"human": "question", "assistant": "answer"}
        
        Returns:
            Dictionnaire avec les champs mappés
        """
        mapped = {}
        for template_field, original_field in field_mapping.items():
            if original_field not in data:
                raise ValueError(f"Champ '{original_field}' manquant dans les données")
            mapped[template_field] = data[original_field]
        return mapped
    
    def format_with_mapping(self, data: Dict, field_mapping: Dict[str, str]) -> str:
        """
        Mappe et formate en une seule étape
        
        Args:
            data: Données originales
            field_mapping: Mapping des champs
            
        Returns:
            String formatée
        """
        mapped_data = self.map_fields(data, field_mapping)
        return self.format_single(mapped_data)


class InstructionDatasetLoader:
    """
    Charge des datasets d'instruction depuis différents formats
    """
    
    @staticmethod
    def load_json(file_path: str) -> List[Dict]:
        """
        Charge un fichier JSON
        Format attendu: liste de dictionnaires ou dictionnaire avec une clé "data"
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'data' in data:
            return data['data']
        else:
            raise ValueError("Format JSON invalide. Attendu: liste ou dict avec clé 'data'")
    
    @staticmethod
    def load_jsonl(file_path: str) -> List[Dict]:
        """
        Charge un fichier JSONL (une ligne JSON par exemple)
        """
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    
    @staticmethod
    def load_csv(file_path: str) -> List[Dict]:
        """
        Charge un fichier CSV
        La première ligne doit contenir les noms de colonnes
        """
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(dict(row))
        return data
    
    @staticmethod
    def load_dataset(file_path: str) -> List[Dict]:
        """
        Détecte automatiquement le format et charge le dataset
        
        Args:
            file_path: Chemin vers le fichier
            
        Returns:
            Liste de dictionnaires
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Fichier non trouvé: {file_path}")
        
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.json':
            return InstructionDatasetLoader.load_json(file_path)
        elif ext == '.jsonl':
            return InstructionDatasetLoader.load_jsonl(file_path)
        elif ext == '.csv':
            return InstructionDatasetLoader.load_csv(file_path)
        else:
            raise ValueError(f"Format de fichier non supporté: {ext}")
    
    @staticmethod
    def save_json(data: List[Dict], file_path: str, indent: int = 2):
        """Sauvegarde en JSON"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
    
    @staticmethod
    def save_jsonl(data: List[Dict], file_path: str):
        """Sauvegarde en JSONL"""
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')


class InstructionTuningPipeline:
    """
    Pipeline complet pour l'instruction tuning
    Combine chargement, formatage et préparation des données
    """
    
    def __init__(
        self,
        template: Optional[InstructionTemplate] = None,
        field_mapping: Optional[Dict[str, str]] = None
    ):
        """
        Args:
            template: Template d'instruction à utiliser
            field_mapping: Mapping optionnel des champs
        """
        self.formatter = InstructionDataFormatter(template=template)
        self.field_mapping = field_mapping or {}
        self.stats = {
            "total_loaded": 0,
            "total_formatted": 0,
            "errors": 0
        }
    
    def process_file(
        self,
        file_path: str,
        output_path: Optional[str] = None,
        validate: bool = True
    ) -> List[str]:
        """
        Traite un fichier de dataset complet
        
        Args:
            file_path: Chemin vers le fichier source
            output_path: Chemin optionnel pour sauvegarder les données formatées
            validate: Valider que tous les champs requis sont présents
            
        Returns:
            Liste de strings formatées
        """
        # Charger les données
        data = InstructionDatasetLoader.load_dataset(file_path)
        self.stats["total_loaded"] = len(data)
        
        # Formater
        formatted = []
        for i, item in enumerate(data):
            try:
                if self.field_mapping:
                    text = self.formatter.format_with_mapping(item, self.field_mapping)
                else:
                    text = self.formatter.format_single(item)
                formatted.append(text)
            except (ValueError, KeyError) as e:
                self.stats["errors"] += 1
                print(f"⚠️ Erreur ligne {i+1}: {e}")
                if validate:
                    raise
        
        self.stats["total_formatted"] = len(formatted)
        
        # Sauvegarder si demandé
        if output_path:
            formatted_data = [{"text": text} for text in formatted]
            InstructionDatasetLoader.save_jsonl(formatted_data, output_path)
            print(f"✅ Données formatées sauvegardées: {output_path}")
        
        return formatted
    
    def process_list(self, data_list: List[Dict]) -> List[str]:
        """
        Traite une liste de dictionnaires directement
        
        Args:
            data_list: Liste de dictionnaires
            
        Returns:
            Liste de strings formatées
        """
        self.stats["total_loaded"] = len(data_list)
        
        formatted = []
        for item in data_list:
            try:
                if self.field_mapping:
                    text = self.formatter.format_with_mapping(item, self.field_mapping)
                else:
                    text = self.formatter.format_single(item)
                formatted.append(text)
            except (ValueError, KeyError):
                self.stats["errors"] += 1
        
        self.stats["total_formatted"] = len(formatted)
        return formatted
    
    def get_stats(self) -> Dict:
        """Retourne les statistiques du pipeline"""
        return self.stats.copy()


def convert_to_instruction_format(
    data: List[Dict],
    template_name: str = "chat_bot",
    field_mapping: Optional[Dict[str, str]] = None
) -> List[Dict]:
    """
    Fonction utilitaire pour convertir rapidement des données
    
    Args:
        data: Liste de dictionnaires
        template_name: Nom du template ("chat_bot", "alpaca", "qa", etc.)
        field_mapping: Mapping optionnel des champs
        
    Returns:
        Liste de dictionnaires avec le champ "text" formaté
    """
    # Sélectionner le template
    templates_map = {
        "chat": InstructionTemplates.get_chat_template(),
        "chat_bot": InstructionTemplates.get_chat_bot_template(),
        "alpaca": InstructionTemplates.get_alpaca_template(),
        "alpaca_no_input": InstructionTemplates.get_alpaca_no_input_template(),
        "qa": InstructionTemplates.get_qa_template(),
        "completion": InstructionTemplates.get_completion_template(),
        "vicuna": InstructionTemplates.get_vicuna_template(),
        "llama2": InstructionTemplates.get_llama2_template(),
    }
    
    template = templates_map.get(template_name)
    if not template:
        raise ValueError(f"Template inconnu: {template_name}")
    
    pipeline = InstructionTuningPipeline(
        template=template,
        field_mapping=field_mapping
    )
    
    formatted_texts = pipeline.process_list(data)
    
    # Retourner avec les données originales + texte formaté
    result = []
    for original, formatted_text in zip(data, formatted_texts):
        item = original.copy()
        item["formatted_text"] = formatted_text
        result.append(item)
    
    return result


if __name__ == "__main__":
    print("="*60)
    print("MODULE D'INSTRUCTION TUNING")
    print("="*60)
    
    # Exemple d'utilisation
    sample_data = [
        {"human": "Bonjour", "assistant": "Bonjour ! Comment puis-je vous aider ?"},
        {"human": "Quelle est la capitale de la France ?", "assistant": "La capitale de la France est Paris."},
        {"human": "Explique-moi la photosynthèse", "assistant": "La photosynthèse est le processus par lequel les plantes convertissent la lumière du soleil en énergie."}
    ]
    
    print("\n📝 Exemple de données:")
    print(json.dumps(sample_data[0], indent=2, ensure_ascii=False))
    
    # Test avec différents templates
    print("\n" + "="*60)
    print("FORMATAGE AVEC DIFFÉRENTS TEMPLATES")
    print("="*60)
    
    templates_to_test = [
        ("chat_bot", None),
        ("qa", {"question": "human", "answer": "assistant"}),
        ("alpaca_no_input", {"instruction": "human", "output": "assistant"}),
    ]
    
    for template_name, mapping in templates_to_test:
        print(f"\n📌 Template: {template_name}")
        print("-" * 40)
        formatted = convert_to_instruction_format(
            sample_data[:1],
            template_name=template_name,
            field_mapping=mapping
        )
        print(formatted[0]["formatted_text"])
    
    print("\n" + "="*60)
    print("✅ MODULE INSTRUCTION TUNING PRÊT!")
    print("="*60)
