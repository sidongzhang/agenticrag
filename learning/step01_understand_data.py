"""Step 01: follow one financial multi-hop QA sample end to end.

This script is deliberately read-only.  It does not call an LLM, download a
model, or rebuild an index.  It answers the first practical question in this
project: what exactly is one training sample, and how does it connect back to
the source documents and an Agent execution trace?
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import exact_match, f1_score  # noqa: E402


DATA_DIR = PROJECT_ROOT / "data" / "financial_eval"
CORPUS_PATH = PROJECT_ROOT / "data" / "financial_all" / "corpus_all.json"
TRAIN_PATH = DATA_DIR / "train_qa_pairs_zh_clean.json"
TEST_PATH = DATA_DIR / "qa_pairs_zh_clean.json"
TRACE_PATH = DATA_DIR / "traces_oracle_zh.jsonl"


def load_json(path: Path):
    """Load a normal JSON file."""
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def load_first_trace_for(question: str) -> dict:
    """JSONL stores one JSON object per line; find the matching question."""
    with TRACE_PATH.open(encoding="utf-8") as file:
        for line in file:
            trace = json.loads(line)
            if trace["question"] == question:
                return trace
    raise ValueError("No Oracle trace matched the selected question")


def shorten(text: str, limit: int = 260) -> str:
    """Make long PDF chunks readable in terminal output."""
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[:limit] + "..."


def main() -> None:
    train_samples = load_json(TRAIN_PATH)
    test_samples = load_json(TEST_PATH)
    corpus = load_json(CORPUS_PATH)

    # A dictionary lets us locate source text by chunk_id in constant time.
    corpus_by_id = {chunk["chunk_id"]: chunk for chunk in corpus}

    print("\n=== 1. We first confirm what data really exists ===")
    print(f"Corpus chunks: {len(corpus)}")
    print(f"Training questions: {len(train_samples)}")
    print(f"Test questions: {len(test_samples)}")
    print("Training hop distribution:", dict(Counter(x["hop_count"] for x in train_samples)))
    print("Training QA types:", dict(Counter(x["qa_type"] for x in train_samples)))

    # Use a stable sample ID so everyone sees the same example on every run.
    sample = next(item for item in train_samples if item["id"] == "mhop_000001")

    print("\n=== 2. The final question cannot be answered by one lookup ===")
    print("Question:", sample["final_question"])
    print("Gold answer:", sample["final_answer"])
    print(f"Type: {sample['qa_type']}; required hops: {sample['hop_count']}")

    print("\n=== 3. Each hop is a smaller, searchable question ===")
    for hop in sample["hops"]:
        chunk = corpus_by_id[hop["doc_chunk_id"]]
        print(f"\nHop {hop['hop_idx']}")
        print("Sub-question:", hop["question"])
        print("Expected fact:", hop["answer"])
        print("Gold chunk_id:", hop["doc_chunk_id"])
        print("Source report:", chunk["title"])
        print("Source excerpt:", shorten(chunk["text"]))
        print("Retrievers known to reach it:", hop.get("search_tools") or "not annotated at seed stage")

    print("\n=== 4. Oracle trace shows the ideal Agent execution ===")
    trace = load_first_trace_for(sample["final_question"])
    for event in trace["trace"]:
        print(event)

    print("\n=== 5. Evaluation compares prediction with the gold answer ===")
    prediction = trace["pred"]
    aliases = sample.get("answer_aliases", [])
    print("Prediction:", prediction)
    print("Exact Match:", exact_match(prediction, sample["final_answer"], aliases))
    print("Token F1:", f1_score(prediction, sample["final_answer"], aliases))

    print("\nStep 01 complete: question -> hops -> source chunks -> Agent trace -> metrics")


if __name__ == "__main__":
    main()
