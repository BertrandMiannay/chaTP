# ChaTP — Benchmark d'approches IA pour la plongée

Chat IA répondant à des questions sur les réglementations de plongée (FFESSM), avec benchmark de plusieurs approches.

## Approches

- **Brute force** : tous les PDFs extraits en texte et envoyés en contexte à chaque question
- **Step-by-step** : résumés JSON générés au démarrage, un LLM filtre les documents pertinents, un second répond avec les seuls documents sélectionnés
- **RAG** : embeddings + recherche vectorielle (ChromaDB)

## Résultats du benchmark

Questions posées sur la réglementation FFESSM, scorées par un LLM-as-judge (1–5).

| Approche     | Moy. tokens | Moy. latence | Score moyen |
|--------------|-------------|--------------|-------------|
| Brute Force  | 218 638      | 9 368 ms     | 4,3 / 5     |
| Step by Step | 123 479      | 6 804 ms     | 4,3 / 5     |
| RAG          | 4 558        | 3 036 ms     | 3,8 / 5     |

**Observations :**
- Brute Force et Step by Step atteignent la même qualité de réponse (4,3/5), mais Step by Step divise la consommation de tokens par ~1,8 et réduit la latence de 27 %.
- RAG réduit drastiquement les tokens (×48 vs Brute Force) et la latence (×3), au prix d'une légère perte de qualité (−0,5 point).

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

## Todo

- [ ] Évaluation manuelle des questions
- [ ] Ajouter des questions plus transversales
