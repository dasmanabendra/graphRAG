# GraphRAG vs Vector RAG

A small, from-scratch comparison of GraphRAG and vector RAG on a HotpotQA subset, built to see
where graph-based retrieval actually helps (and where it doesn't) versus plain vector similarity
search. Both pipelines are hand-built (no Microsoft GraphRAG, LlamaIndex, or LangChain graph
tooling) so every stage is inspectable.

## Setup

```bash
pip install -r requirements.txt
```

Needs a `GEMINI_API_KEY`. Either set it in `.env` (copy `.env.example`), or store it in Windows
Credential Manager as a Generic Credential with target `gemini_api` / username `gemini_api` --
`config.py` checks the env var first, then falls back to Credential Manager automatically.

## Run order

```bash
python scripts/00_smoke_test.py        # confirms API key + model IDs work
python scripts/01_fetch_data.py --n 25 # fetch HotpotQA subset, build deduped corpus
python scripts/02_build_vector_index.py
python scripts/03_build_graph_index.py # the expensive step: extraction + community summaries
python scripts/04_run_eval.py          # side-by-side EM/F1/retrieval/call-count report
python scripts/inspect_graph.py        # free -- node/edge/community stats, no LLM calls
streamlit run app.py                   # interactive Q&A comparison UI
```

All LLM/embedding calls are logged to `data/call_log.jsonl` and response-cached in
`data/response_cache/` (keyed by prompt+model+schema), so re-running any script after a bug fix
doesn't re-spend the daily call budget for prompts that haven't changed.

## Known limitations

- **Entity resolution** is pure string matching (exact, then `difflib` fuzzy match), scoped to
  same-type entities only. No embedding- or LLM-based dedup, so near-miss surface forms across
  different types (rare) won't merge, and no LLM call is spent on it.
- **Community fragmentation**: HotpotQA's 25 questions are mostly independent topic clusters, so
  Louvain produces many singleton/pair "communities" (entities with no extracted relation to
  anything else in the corpus). Only communities of 3+ entities get an LLM summary; smaller ones
  get a cheap templated description instead.
- **Cache determinism across process restarts**: community summarization prompts are now sorted
  deterministically, but NetworkX's Louvain implementation still uses hash-randomized internals
  in a couple of places, so the exact community *partition* (not just ordering) can vary slightly
  between fresh process runs -- occasionally causing more cache misses than expected on a full
  re-index. Doesn't affect correctness, just occasionally costs a few more calls than the minimum.
- **Free-tier rate limits are real**: both the chat and embedding APIs enforce per-minute caps
  tighter than the batch sizes here can avoid. `common/retry.py` handles 429s with backoff
  automatically; expect occasional 15-60s pauses during indexing.
