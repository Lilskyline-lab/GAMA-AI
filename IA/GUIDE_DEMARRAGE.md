# Guide de Démarrage Rapide - Système d'Instruction Tuning

## ✅ Système Opérationnel !

Votre système d'instruction tuning pour LLM est maintenant **entièrement configuré et fonctionnel**.

## 🎯 Fonctionnement Automatique

**Vous n'avez RIEN à faire pendant l'entraînement** - le système fonctionne de manière complètement automatique :

1. ✅ Les données sont automatiquement chargées depuis 3 sources
2. ✅ L'instruction tuning est appliqué automatiquement  
3. ✅ Le formatage se fait selon le template choisi
4. ✅ L'entraînement démarre et se termine seul
5. ✅ Le modèle est sauvegardé automatiquement

## 📁 Structure du Projet

```
.
├── data/                          # VOS DONNÉES ICI
│   ├── example_instructions.json  # Exemples fournis (10 Q&A)
│   └── README.md                  # Documentation des formats
├── training/                      # Code d'entraînement
│   ├── TrainWithRealDataSet.py   # Script principal (DÉJÀ CONFIGURÉ)
│   ├── Model/                     # Architecture GPT-2
│   └── Tokenizer/                 # Tokenizer BPE
├── utils/                         # Outils
│   └── instruction_tuning.py     # Module d'instruction tuning
└── saved_models/                  # Modèles entraînés
    └── my_llm/                    # Votre modèle
```

## 🚀 Comment Utiliser

### 1. Ajouter Vos Propres Données (Optionnel)

Créez simplement un fichier dans `data/` :

**Format JSON :**
```json
[
  {
    "human": "Votre question",
    "assistant": "La réponse"
  }
]
```

**Format JSONL :**
```jsonl
{"human": "Question 1", "assistant": "Réponse 1"}
{"human": "Question 2", "assistant": "Réponse 2"}
```

**Format CSV :**
```csv
human,assistant
"Question 1","Réponse 1"
"Question 2","Réponse 2"
```

### 2. Démarrer l'Entraînement

Le workflow est déjà configuré et s'exécute automatiquement !

Pour relancer manuellement :
```bash
python training/TrainWithRealDataSet.py
```

### 3. Modifier les Paramètres (Optionnel)

Éditez `training/TrainWithRealDataSet.py`, ligne ~795 :

```python
trainer.train_one_cycle(
    num_articles=5,          # Nombre d'articles Wikipedia
    qa_per_article=3,        # Q&A par article
    num_dialogues=30,        # Dialogues OASST1
    epochs=2,                # Époques d'entraînement
    batch_size=4,            # Taille du batch
    lr=5e-5,                 # Learning rate
    use_custom_data=True     # Utiliser vos données
)
```

### 4. Changer le Template d'Instruction (Optionnel)

Ligne ~764 dans le même fichier :

```python
trainer = ContinuousTrainer(
    # ...
    instruction_template="chat_bot"  # Changez ici
)
```

**Templates disponibles :**
- `chat_bot` (défaut) : `Human: {question}\nBot: {answer}`
- `alpaca` : Format Alpaca standard
- `qa` : `Question: {q}\nAnswer: {a}`
- `llama2` : Format Llama 2 Chat
- `vicuna` : Format Vicuna

## 📊 Résultats du Premier Entraînement

✅ **Entraînement terminé avec succès !**

- **96 exemples** formatés avec instruction tuning
  - 30 exemples personnalisés (data/example_instructions.json)
  - 42 paires Q&A depuis Wikipedia (5 articles)
  - 24 dialogues depuis OASST1
- **2 époques** complétées
- **Loss** : 7.54 → 6.56 (diminution de 13%)
- **Modèle sauvegardé** : `saved_models/my_llm/model.pt`

## 🎓 Sources de Données

### 1. Données Personnalisées
- Fichiers dans `data/`
- Formats : JSON, JSONL, CSV
- **Automatiquement détectés et chargés**

### 2. Wikipedia
- Articles aléatoires en français (ou anglais)
- Catégorisation automatique (science, histoire, géographie, etc.)
- Génération de Q&A contextuelles

### 3. OASST1 (Hugging Face)
- Dataset de dialogues conversationnels
- 84,437 conversations disponibles
- Filtrage par langue automatique

## 🔧 Commandes Utiles

**Voir les statistiques :**
Le système affiche automatiquement les stats après chaque cycle.

**Relancer l'entraînement :**
```bash
python training/TrainWithRealDataSet.py
```

**Tester le module d'instruction tuning :**
```bash
python utils/instruction_tuning.py
```

## 💡 Prochaines Étapes Suggérées

1. **Ajoutez vos propres données** dans `data/`
2. **Augmentez les époques** pour un meilleur entraînement
3. **Testez différents templates** d'instruction
4. **Implémentez LoRA/QLoRA** pour un fine-tuning plus efficace
5. **Ajoutez un système d'évaluation** automatique

## 📝 Notes Importantes

- ⚡ **Aucune intervention manuelle nécessaire** pendant l'entraînement
- 🔄 Les données sont répétées 3× pour renforcer l'apprentissage
- 💾 Le modèle est sauvegardé automatiquement après chaque cycle
- 📊 L'historique d'entraînement est conservé dans `saved_models/my_llm/training_history.json`
- 🗂️ Les sujets traités sont trackés dans `saved_models/my_llm/trained_topics.json`

## ❓ Questions Fréquentes

**Q: Puis-je utiliser mes propres données uniquement ?**
R: Oui ! Mettez `use_custom_data=True` et `num_articles=0, num_dialogues=0`

**Q: Comment changer la langue ?**
R: Modifiez `language='fr'` en `language='en'` dans le ContinuousTrainer

**Q: Où est sauvegardé le modèle ?**
R: Dans `saved_models/my_llm/model.pt`

**Q: Comment voir les logs d'entraînement ?**
R: Ils s'affichent automatiquement dans la console pendant l'exécution

## 🎉 Félicitations !

Votre système d'instruction tuning est **100% opérationnel** et **entièrement automatisé** !
