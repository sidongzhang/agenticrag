import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = PROJECT_ROOT / "data" / "financial_all" / "corpus_all.json"

print(CORPUS_PATH)

with CORPUS_PATH.open("r", encoding="utf-8") as file:
    corpus = json.load(file)

print(type(corpus))
print(len(corpus))
print(corpus[0].keys())