# Configuration des Datasets pour Fine-tuning LoRA

## 📋 Vue d'ensemble du projet

**Objectif :** Fine-tuner un modèle de langage en LoRA pour :
- Conversations naturelles (français/anglais)
- Function calling (requêtes HTTP)
- Recherche d'informations externes

---

## 🎯 Datasets pour le Fine-tuning

### 1. Conversations générales - Qualité Premium

#### **Anthropic/hh-rlhf**
- **Langue :** Anglais
- **Taille :** ~160k conversations
- **Usage :** Conversations helpful & harmless de haute qualité
- **Pourquoi :** Ton naturel, empathique et sécurisé

#### **HuggingFaceH4/ultrachat_200k**
- **Langue :** Anglais
- **Taille :** 200k conversations
- **Usage :** Diversité de sujets conversationnels
- **Pourquoi :** Excellente qualité synthétique, large couverture

#### **OpenAssistant/oasst2**
- **Langue :** Multilingue (anglais + français)
- **Taille :** ~160k messages
- **Usage :** Conversations multi-tours validées par la communauté
- **Pourquoi :** Données réelles, votes de qualité

### 2. Instructions en français

#### **bofenghuang/vigogne-instruction-following-v1.0**
- **Langue :** Français
- **Taille :** ~50k instructions
- **Usage :** Renforcer les capacités en français
- **Pourquoi :** Meilleur dataset d'instructions françaises

### 3. Function Calling & Tool Use

#### **xlangai/xlam-function-calling-60k**
- **Langue :** Anglais
- **Taille :** 60k exemples
- **Usage :** Apprendre à identifier et structurer les appels de fonctions
- **Pourquoi :** Dataset le plus complet pour function calling

#### **glaiveai/glaive-function-calling-v2**
- **Langue :** Anglais
- **Taille :** ~110k exemples
- **Usage :** Compléter l'apprentissage des function calls
- **Pourquoi :** Diversité de fonctions et scénarios réels

#### **Nexusflow/NexusRaven-V2-data**
- **Langue :** Anglais
- **Taille :** Variable
- **Usage :** Spécialisation tool use avancé
- **Pourquoi :** Focus sur l'exécution précise d'outils

---

## 🔤 Dataset pour l'entraînement du Tokenizer

### **allenai/c4**
- **Langue :** Multilingue (c4-en + c4-fr)
- **Taille :** ~300+ Go par langue
- **Usage :** Entraîner un tokenizer BPE à 50k vocab
- **Pourquoi :** 
  - Corpus web massif et diversifié
  - Nettoyé et filtré pour la qualité
  - Vocabulaire représentatif du web réel
  - Inclut français et anglais

**Portions recommandées :**
- 60% C4 anglais (c4-en)
- 40% C4 français (c4-fr)

---

## 📊 Répartition recommandée du fine-tuning

```
Conversations générales (70%)
├── ultrachat_200k          → 30%
├── Anthropic/hh-rlhf       → 25%
├── oasst2                  → 10%
└── vigogne (français)      → 5%

Function Calling (30%)
├── xlam-function-calling   → 15%
├── glaive-function-calling → 10%
└── NexusRaven-V2           → 5%
```

---

## 🛠️ Pipeline d'entraînement

### Étape 1 : Tokenizer
1. Télécharger échantillons de `allenai/c4` (en + fr)
2. Entraîner tokenizer BPE 50k vocab
3. Tokens spéciaux : `<pad>`, `<s>`, `</s>`, `<unk>`, `<search>`, `</search>`

### Étape 2 : Fine-tuning LoRA
1. Charger tous les datasets listés ci-dessus
2. Normaliser au format conversations uniforme
3. Mélanger selon les proportions recommandées
4. Fine-tuner avec LoRA (rank 16-64)

---

## ⚙️ Paramètres suggérés

**Tokenizer :**
- Vocab size : 50,000
- Algorithm : BPE (Byte-Pair Encoding)
- Training data : 10-50 Go de C4 (mix en/fr)

**LoRA Fine-tuning :**
- Rank (r) : 32-64
- Alpha : 64-128
- Dropout : 0.05-0.1
- Target modules : q_proj, v_proj, k_proj, o_proj
- Epochs : 1-3 selon la taille totale

---

## 📝 Notes importantes

- **Filtrage :** Filtrer oasst2 pour garder uniquement français/anglais
- **Qualité :** Vérifier et nettoyer les datasets avant entraînement
- **Équilibrage :** Ajuster les proportions selon vos besoins spécifiques
- **Validation :** Garder 5-10% de chaque dataset pour validation

---

## 🔗 Liens Hugging Face

- Anthropic/hh-rlhf : `https://huggingface.co/datasets/Anthropic/hh-rlhf`
- ultrachat_200k : `https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k`
- oasst2 : `https://huggingface.co/datasets/OpenAssistant/oasst2`
- vigogne : `https://huggingface.co/datasets/bofenghuang/vigogne-instruction-following-v1.0`
- xlam-function-calling : `https://huggingface.co/datasets/xlangai/xlam-function-calling-60k`
- glaive-function-calling : `https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2`
- NexusRaven : `https://huggingface.co/datasets/Nexusflow/NexusRaven-V2-data`
- C4 : `https://huggingface.co/datasets/allenai/c4`

---

**Date de création :** 27 octobre 2025  
**Version :** 1.0