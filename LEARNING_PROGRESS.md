# AgenticRAG learning progress

This file records what has actually been run and understood.  A step is marked
complete only after its command succeeds on this machine.

## Current status

- [x] Step 01 — Trace one multi-hop sample from question to source chunks,
  Oracle Agent execution, and EM/F1 evaluation.
- [ ] Step 02 — Run BM25 keyword retrieval and inspect why it succeeds or fails.
- [ ] Step 03 — Compare keyword, semantic, graph, and hybrid retrieval.
- [ ] Step 04 — Run one PEV Agent flow and inspect shared state at every node.
- [ ] Step 05 — Evaluate a small batch and diagnose failures by hop.
- [ ] Step 06 — Rebuild a small slice of the financial data pipeline.
- [ ] Step 07 — Convert Oracle traces into ReAct SFT messages.
- [ ] Step 08 — Run and evaluate Qwen3-4B LoRA SFT on a GPU machine.
- [ ] Step 09 — Inspect multi-turn verl rollout and reward calculation.
- [ ] Step 10 — Run GRPO experiments and compare them with the SFT baseline.

## Environment confirmed

- Machine: Apple Silicon macOS (`arm64`)
- Project Python: `.venv/bin/python`, Python 3.11
- Original CUDA/vLLM/verl training commands require a Linux NVIDIA GPU machine;
  CPU-compatible data, retrieval, metrics, and orchestration work will be done
  locally before transferring the same repository and artifacts for training.

## Commands completed

```bash
cd <repo-root>
.venv/bin/python learning/step01_understand_data.py
```

## Evidence from Step 01

- Financial corpus: 1,933 chunks
- Training set: 797 multi-hop questions
- Test set: 185 multi-hop questions
- The inspected sample links each hop to a real `doc_chunk_id` and source text.
- Its Oracle trace contains Router, Planner, Executor, Verifier, and Synthesizer
  events and receives EM=1 and F1=1.
