# MCD → MLD Transformer · Système IA Merise

[![Hugging Face Spaces](https://img.shields.io/badge/Demo-HuggingFace-yellow)](https://huggingface.co/spaces/237-zla/mcd-mld)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org)

Système hybride de transformation automatique du Modèle Conceptuel de Données
(MCD/Merise) vers le Modèle Logique de Données (MLD) assisté par Intelligence
Artificielle.

---

## Présentation

Ce projet automatise la transformation MCD → MLD selon la méthodologie Merise,
en combinant :
- Un **moteur de règles déterministe** (4 règles Merise)
- Un **modèle Random Forest** pour la classification des cardinalités
- Un **générateur SQL DDL** automatique

Développé dans le cadre du cours de Bases de Données — SDIA Licence 3,
École Nationale Supérieure Polytechnique de Douala (ENSPD), Cameroun.

---

## Architecture
MCD (JSON)

│

▼

[Dataset synthétique 3000 exemples]

│

▼

[Random Forest — classification cardinalités]

│

▼

[Moteur de règles Merise — 4 règles]

│

▼

MLD (notation Merise) + SQL DDL

## Règles Merise implémentées

| Règle | Condition | Action |
|-------|-----------|--------|
| 1 | Toute entité | → Table avec clé primaire |
| 2 | Association 1:N | → FK migre vers l'entité enfant |
| 3 | Association N:M | → Table de jonction + PK composite |
| 4 | Association 1:1 | → FK vers l'entité optionnelle |

---

## Format d'entrée (JSON)

```json
{
  "entites": [
    {
      "nom": "CLIENT",
      "attributs": ["id_client", "nom", "email"],
      "cle": "id_client"
    }
  ],
  "associations": [
    {
      "nom": "commande",
      "entites": ["CLIENT", "COMMANDE"],
      "cardinalites": ["1,N", "1,1"],
      "attributs": []
    }
  ]
}
```

---

## Installation locale

```bash
git clone https://github.com/237-zla/mcd-to-mld
cd mcd-to-mld
pip install -r requirements.txt

# Interface web locale (Flask + Tkinter)
python launcher.py

# Interface Gradio locale
python app.py
```

---

## Démo en ligne

[huggingface.co/spaces/237-zla/mcd-mld](https://huggingface.co/spaces/237-zla/mcd-mld)

---

## Structure du projet
mcd-to-mld/

├── dataset_generator.py   # Génération dataset synthétique

├── ml_model.py            # Modèle Random Forest

├── rules_engine.py        # Moteur de règles Merise

├── mld_generator.py       # Générateur MLD + SQL

├── app.py                 # Interface Gradio (HF Space)

├── flask_app.py           # Backend Flask (local)

├── launcher.py            # Lanceur Tkinter (local)

└── templates/

└── index.html         # Interface web locale

---

## Auteur

**OURBANI DOUBLA MAMOUDOU Ibrahima** — SDIA L3, ENSPD Douala, Cameroun  
GitHub : [@237-zla](https://github.com/237-zla)

---

## Licence

MIT License — libre d'utilisation, modification et distribution.
