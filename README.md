# Mini projet RAG (Python)

Un petit exemple de RAG (Retrieval Augmented Generation) 100% local avec TF-IDF. Il charge des documents simples, construit un index, retrouve les passages pertinents et genere une reponse extractive.

## Structure

- `main.py`: point d entree.
- `rag.py`: logique RAG (chargement, chunking, index, retrieval, reponse).
- `data/`: documents sources.
- `test.py`: test minimal.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Execution

```bash
python main.py "What is RAG?"
```

## Test

```bash
python test.py
```

## Etapes cle (resume)

1. Charger les documents locaux dans `data/`.
2. Decouper en chunks et construire un index TF-IDF.
3. Vectoriser la requete et calculer la similarite cosine.
4. Recuperer les meilleurs chunks et generer une reponse extractive.

