# ChaTP — Benchmark d'approches IA pour la plongée

Chat IA répondant à des questions sur les réglementations de plongée (FFESSM), avec benchmark de plusieurs approches.

## Approches

- **Brute force** : tous les PDFs extraits en texte et envoyés en contexte à chaque question
- **Step-by-step** : résumés JSON générés au démarrage, un LLM filtre les documents pertinents, un second répond avec les seuls documents sélectionnés
- **RAG** *(à venir)* : embeddings + recherche vectorielle

## Prérequis

- Python 3.10+
- [Poetry](https://python-poetry.org/)
- Une clé API Mistral

## Installation

```bash
poetry install
```

## Configuration

Créer un fichier `.env` à la racine :

```env
MISTRAL_API_KEY=...
```

## Lancer l'interface

```bash
poetry run streamlit run ui/app.py
```

## Interface

### Onglet Chat

Posez des questions sur la réglementation de plongée. Un sélecteur permet de choisir l'approche utilisée pour répondre (Brute Force ou Step by Step). Les métriques (tokens, latence) sont affichées pour chaque échange.

### Onglet Benchmark

Lance une série de questions prédéfinies (`benchmark/questions.json`) sur les approches sélectionnées et compare :
- Tokens consommés
- Latence
- Score de qualité via LLM-as-judge (optionnel, 1-5)

## Structure

```
chaTP/
├── api/
│   ├── base.py                  # BaseAPI + APIResponse
│   ├── claude_api.py            # Implémentation Claude
│   └── mistral_api.py           # Implémentation Mistral
├── approaches/
│   ├── base.py                  # BaseApproach + ApproachResponse
│   ├── utils.py                 # Utilitaires partagés (extraction PDF)
│   ├── brute_force/
│   │   └── approach.py
│   └── step_by_step/
│       └── approach.py          # Résumés JSON + filtre + réponse
├── benchmark/
│   ├── questions.json           # Questions de test (question, expected_answer, short_label)
│   ├── judge.py                 # LLM-as-judge (score 1-5)
│   └── runner.py                # Orchestration du benchmark
├── ui/
│   └── app.py                   # Interface Streamlit
└── data/                        # PDFs réglementaires
```

## Ajouter une nouvelle API

Créer une classe héritant de `BaseAPI` dans `api/` et implémenter `send()` :

```python
from api.base import BaseAPI, APIResponse

class MyAPI(BaseAPI):
    def send(self, messages: list[dict], system: str | None = None) -> APIResponse:
        ...
```

## Ajouter une nouvelle approche

Créer un dossier dans `approaches/`, hériter de `BaseApproach` et implémenter `ask()` :

```python
from approaches.base import BaseApproach, ApproachResponse

class MyApproach(BaseApproach):
    name = "my_approach"

    def ask(self, question: str) -> ApproachResponse:
        ...
```

Puis l'ajouter au dict `APPROACHES` dans `ui/app.py`.
