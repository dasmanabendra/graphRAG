# GraphRAG Field Notes

*What a from-scratch GraphRAG, Vector RAG, and Hybrid RAG pipeline actually taught me — the fundamentals, the bugs I found in my own code, and the numbers that settled the arguments.*

**Corpus:** 249 Wikipedia paragraphs (HotpotQA) · **Eval set:** 25 questions · **Models:** `gemini-3.1-flash-lite`, `gemini-embedding-001` · **Pipelines:** Vector · Graph · Hybrid (RRF)

> Personal reference — learned by building it, not by reading about it. Not a tutorial; a page to come back to and recollect what building this taught me.

## Contents

0. [Quick reference](#0-quick-reference)
1. [GraphRAG fundamentals](#1-graphrag-fundamentals)
2. [How this project implemented it](#2-how-this-project-implemented-it)
3. [Bugs found, with real examples](#3-bugs-found-with-real-examples)
4. [Metrics: how "better" was measured](#4-metrics-how-better-was-measured)
5. [Head-to-head results](#5-head-to-head-results)
6. [Hybrid retrieval in action](#6-hybrid-retrieval-in-action)
7. [Toy vs. production-grade GraphRAG](#7-toy-vs-production-grade-graphrag)
8. [Glossary](#8-glossary)
9. [Appendix: project file map](#9-appendix-project-file-map)

---

## 0. Quick reference

Read this section and stop, unless you need the reasoning behind it. Everything past this point exists to justify these tables.

**Eval scoreboard (EM):** Vector RAG `0.56` · GraphRAG `0.48` · **Hybrid RAG `0.60`**

Hybrid won not by being a cleverer retriever, but by *arithmetically outvoting* GraphRAG's noise: fusing in the vector list's clean top-5 pushed the graph pipeline's hub-node contamination ([§3](#3-bugs-found-with-real-examples)) out of the final context often enough to close its recall gap without losing GraphRAG's multi-hop reach.

### A. Preprocessing — building the index, once

**GraphRAG — 5 steps**

| Step | What it does | Cost |
|---|---|---|
| Extraction | LLM reads each chunk batch, pulls out entities + relations as structured JSON | ~36 LLM calls |
| Entity resolution | Merges surface forms that name the same real-world thing (exact + fuzzy string match) | free |
| Graph construction | Builds the node/edge graph from resolved entities and relations | free |
| Community detection | Louvain clustering finds densely-connected subgroups of nodes | free |
| Community summarization | LLM summarizes clusters of 3+ nodes; smaller clusters get a template stub | 79 LLM calls + 492 free stubs |

**Vector RAG — 1 step**

| Step | What it does | Cost |
|---|---|---|
| Embed & index | Embeds every chunk, upserts into a Chroma collection | ~16 embed calls |

**Hybrid RAG — 0 steps** — reuses the GraphRAG and Vector RAG indexes unchanged. Nothing to build.

### B. Inference — answering one question

**GraphRAG — 4 steps**

| Step | What it does | Cost |
|---|---|---|
| Entity matching | Finds entities named in the question (substring, then fuzzy fallback) | free |
| Neighbor expansion | Walks 1-hop neighbors of matched entities, ranked by edge weight, capped at 20 | free |
| Context gathering | Collects each focus entity's `chunk_ids` backreference | free |
| Generate answer | LLM answers from graph context + community summaries, reasoning before answer | 1 chat call |

**Vector RAG — 3 steps**

| Step | What it does | Cost |
|---|---|---|
| Embed question | Embeds the question text | 1 embed call |
| Similarity search | Top-5 nearest chunks by distance in Chroma | free |
| Generate answer | LLM answers from the 5 retrieved passages | 1 chat call |

**Hybrid RAG — 4 steps**

| Step | What it does | Cost |
|---|---|---|
| Vector retrieval | Same as Vector RAG's embed + search | 1 embed call |
| Graph retrieval | Same as GraphRAG's match + expand + gather | free |
| Fuse (RRF) | Merges both ranked lists by `1/(60+rank)`, keeps top 6 | free |
| Generate answer | LLM answers from the fused passages + community summary | 1 chat call |

### C. Concept cheat-sheet

*The vocabulary §1 leans on — full definitions in the [glossary](#8-glossary).*

| Term | In one line |
|---|---|
| `entity` | A node in the graph — a person, place, org, or work pulled out of the text |
| `relation / edge` | A connection between two entities, weighted by how often they co-occur |
| `node degree` | How many edges touch a node — a rough proxy for "how central" |
| `community` | A densely-connected cluster of nodes, found by graph clustering |
| `chunk_id` | The source passage a fact came from |
| `local search` | Retrieval that starts from named entities and expands outward |
| `global search` | Retrieval that starts from community summaries, for broad questions |
| `hub node` | An entity mentioned across so many unrelated chunks it leaks noise |
| `bridge node` | An entity that connects two otherwise-separate matched entities |
| `RRF` | Reciprocal Rank Fusion — merges ranked lists by `1/(k+rank)` |
| `EM` | Exact Match — 1 if the normalized answer matches gold exactly, else 0 |
| `F1` | Token-overlap score between predicted and gold answer |

### D. Which pipeline wins, when

*The practical payoff — distilled from [§5](#5-head-to-head-results)'s eval and [§1](#1-graphrag-fundamentals)'s concepts.*

| Question pattern | Best pipeline | Why |
|---|---|---|
| Single-fact lookup ("who directed X") | **Vector RAG** | Dense retrieval already nails it; graph traversal adds hub-node noise for no recall gain |
| Multi-hop, connecting two named entities | **Hybrid RAG** | Graph supplies the connecting edge vector similarity can't see; RRF suppresses the hub-node noise a pure-graph pipeline would keep |
| Broad thematic ("what are the main topics in X") | **Neither, honestly** | This needs global search over community summaries — a mode this project never implemented (see [§7](#7-toy-vs-production-grade-graphrag)) |

---

## 1. GraphRAG fundamentals

Everything here applies to any GraphRAG system, not just this one. This project's numbers show up only as evidence for the concept, not as the subject.

### 1.0 The mental model

> Build a map of entities and how they relate to each other. Then answer a question either by walking outward from a named landmark on that map — **local search** — or by reading the map's regional summaries — **global search**. Everything below is detail behind this one picture.

### 1.1 The problem it solves

Vector similarity search retrieves passages that sound like the question. That works when the answer lives in one passage. It breaks on questions that only resolve by *connecting* two facts that don't share vocabulary — "Were the directors of these two unrelated-sounding films born in the same country?" Neither film's Wikipedia paragraph mentions the other; nothing in the embedding space pulls them into the same neighborhood. The connection exists only in the *relationship* between two entities, not in any single chunk's wording — and that relationship is exactly what a similarity search over chunks has no way to represent.

GraphRAG's answer: extract the entities and relationships explicitly, at index time, into a structure a query can traverse — so a multi-hop question becomes a graph walk instead of a vocabulary-matching problem.

### 1.2 The knowledge graph

An **entity** is a node — a person, place, organization, or work. A **relation** is an edge between two entities, usually directed ("directed by", "located in"), carrying a **weight** (how many times the two entities co-occurred across the corpus) and a **provenance** list (which source chunks the relation was extracted from). A node's **degree** — how many edges touch it — is a rough proxy for how central or well-connected that entity is in the corpus. A handful of high-degree nodes ("United States", "Los Angeles" in a Wikipedia corpus) are usually unavoidable, and, as [§3](#3-bugs-found-with-real-examples) covers, usually a liability.

### 1.3 Building the graph (indexing)

```
Extraction  →  Entity resolution  →  Graph construction  →  Community detection  →  Community summarization
```

**Extraction** is a structured-output LLM call per batch of source text: "list the entities and relations mentioned here." **Entity resolution** is the general coreference/entity-linking problem — "Scott Derrickson" and "Derrickson" need to collapse into one node, but that requires either exact matching (cheap, misses paraphrase) or fuzzy/embedding matching (catches more, risks false merges). **Community detection** is standard graph clustering — Louvain and similar algorithms greedily group nodes to maximize *modularity*, the fraction of edges that fall inside clusters versus what you'd expect by chance. **Community summarization** then compresses each cluster into a paragraph an LLM can answer questions from without re-reading every source chunk in it — this is the piece that makes global search (1.4) possible at all.

### 1.4 Querying the graph (retrieval)

A question first has to be **linked** to the graph — which nodes does it name? From there, two general strategies diverge:

| Mode | Starts from | Good for |
|---|---|---|
| **Local search** | Named entities in the question, expanded 1+ hops | Specific, connectable facts ("is X related to Y") |
| **Global search** | Community summaries across the whole graph | Broad, thematic questions with no single entity anchor |
| **Hybrid local+global** | Both, merged (the "DRIFT search" idea) | Questions that need both a specific anchor and broader context |

Note the terminology carefully here: this "hybrid" means combining GraphRAG's own two internal search modes. It is *not* the same "hybrid" as 1.5 below — a genuinely easy pair of concepts to conflate because both are called "hybrid," and this project only ever built the local-search half of this table (see [§7](#7-toy-vs-production-grade-graphrag)).

### 1.5 Vector+graph fusion — a different "hybrid"

Separately from anything internal to GraphRAG, a graph retriever and a dense vector retriever have *complementary* failure modes: vector search is strong on single-passage lookups and weak on multi-hop connections; graph traversal is the reverse. Running both independently and merging their ranked results is a general RAG pattern, not a GraphRAG-specific one.

The merge technique used here — **Reciprocal Rank Fusion (RRF)** — predates GraphRAG entirely; it's a standard technique from search/IR for combining ranked lists without a trained reranker. Each retriever ranks the same candidate item; every item gets a score of `1/(k+rank)` per list it appears in (with a constant `k`, typically 60, that discounts how much any single rank position matters), summed across lists. An item both retrievers agree on outranks one only a single retriever found — pure arithmetic, no extra model call. [Section 6](#6-hybrid-retrieval-in-action) walks through this on a real question.

---

## 2. How this project implemented it

Numbers and outcomes, condensed — each line maps straight back to a concept in §1.3/1.4.

### 2.1 GraphRAG here

Extraction and summarization run on `gemini-3.1-flash-lite`. The 249-chunk corpus resolved to **1,128 entities** and **583 edges** (610 total relation mentions, i.e. some edges were reinforced by more than one co-occurrence). Louvain clustering at the default resolution produced **571 communities** — only **79** of them met the size-3 threshold for an LLM summary; the other **492** got a one-line template stub instead of real summarization. Local search expansion is capped at 20 neighbors and entity matching accepts fuzzy matches above an 0.8 similarity threshold.

### 2.2 Vector RAG here

`gemini-embedding-001` at 768 dimensions, all 249 chunks embedded and upserted into a Chroma collection. Query time retrieves the top 5 nearest chunks by distance — no reranking, no metadata filtering.

### 2.3 Hybrid here

Reuses both indexes completely unchanged — no new preprocessing step exists for hybrid at all. At query time it runs vector retrieval and graph retrieval independently, fuses with RRF (`k=60`, the standard default from the original RRF paper, not tuned for this corpus), and keeps the top 6 fused results (`HYBRID_TOP_K=6`) — also picked, not validated against this corpus size.

---

## 3. Bugs found, with real examples

The hard-won lessons — found by interrogating this project's own output, not by reading about GraphRAG's known failure modes.

> **🐛 Bridge nodes are computed and shown, but never used**
>
> The UI computes and displays "bridge nodes" — entities sitting on a path between two matched entities — on every query. They never feed into retrieval; they're pure observability. Worse, they're structurally silent on the majority of real questions: bridge detection only fires when a question matches *two or more* entities, and most single-entity-anchored questions in this eval set matched exactly one.

> **🐛 Hub-node contamination**
>
> Retrieval reads an entity's raw `chunk_ids` set — every chunk that entity was ever extracted from, corpus-wide, built once at indexing time — instead of walking the graph edges actually near the matched entities. High-degree "hub" entities make this expensive:
>
> - **"Are Local H and For Against both from the United States?"** — matching `United States` pulls in `Brown State Fishing Lake`, a Kansas lake with zero topical connection to either band, purely because both happen to share the "United States" node.
> - **"Were Scott Derrickson and Ed Wood of the same nationality?"** — matching `Los Angeles` pulls in `David Beckham Academy` and `Guns N' Roses discography` — real LA-adjacent things, entirely irrelevant to a nationality question.
>
> Both questions still got the right answer despite the noise ([§4](#4-metrics-how-better-was-measured) covers why that's not the same as the retrieval being correct) — but the leaked chunks were doing nothing useful, and on a harder question they would have crowded out something that mattered.

> **🐛 Community fragmentation**
>
> 571 communities over 1,128 nodes averages out to roughly two nodes per community. Only 79 (14%) cross the size-3 threshold to get a real LLM summary; the remaining 492 (86%) get `"Small cluster: X, Y."` — a template stub with no synthesized meaning. On a corpus this small, single-level Louvain fragments too aggressively for community summaries to carry the weight global search (1.4) needs them to.

---

## 4. Metrics: how "better" was measured

Standard IR/QA evaluation, applied the same way to all three pipelines.

| Metric | Formula | What it tells you |
|---|---|---|
| **EM** | `1 if normalize(pred) == normalize(gold) else 0` | Got the exact answer, after lowercasing / stripping punctuation & articles |
| **F1** | `harmonic_mean(token_precision, token_recall)` | Partial credit for overlapping words when the phrasing differs but the content is close |
| **Retrieval precision** | `\|retrieved ∩ gold_titles\| / \|retrieved\|` | How much of what was fetched was actually relevant |
| **Retrieval recall** | `\|retrieved ∩ gold_titles\| / \|gold_titles\|` | Whether every source needed for the answer was fetched at all |
| **Latency** | wall-clock seconds, end to end | What the user actually waits for |

> **📌 EM and retrieval recall can diverge**
>
> On "Are Local H and For Against both from the United States?", GraphRAG's retrieval precision was a mediocre 0.29 — the hub-node contamination from §3 was actively polluting the context — yet EM was still 1: the correct chunks were *also* present, so the noise didn't change the final answer. A clean EM score on a question like this hides a fragile pipeline; precision is the metric that would have caught it, EM would not.

---

## 5. Head-to-head results

Aggregate scores across the full 25-question eval set, computed from `data/eval_results.json`.

| Pipeline | EM | F1 | Ret. precision | Ret. recall | Latency (mean) |
|---|---:|---:|---:|---:|---:|
| Vector RAG | 0.56 | 0.72 | 0.40 | 1.00 | 1.39 s |
| GraphRAG | 0.48 | 0.59 | 0.25 | 0.78 | 3.24 s |
| **Hybrid RAG** | **0.60** | **0.73** | 0.34 | 1.00 | 2.11 s |

Reading it: Vector RAG's retrieval recall is a perfect 1.00 because top-5 similarity search on a 249-chunk corpus almost always surfaces the right passage somewhere in the five — the corpus is small enough that recall was never really the bottleneck for vector search. GraphRAG's recall drop to 0.78 is the fragmentation problem ([§3](#3-bugs-found-with-real-examples)) showing up numerically: when neighbor expansion misses a relation entirely, nothing else compensates. Hybrid recovers full recall by having vector's clean list to fall back on, while its precision (0.34) still beats plain GraphRAG (0.25) because RRF fusion filters out a share of the graph pipeline's hub-node noise before it reaches the LLM.

Latency is a mean, not a median — both GraphRAG and Hybrid include a handful of rate-limited retries (one question took GraphRAG 40 seconds) that pull their averages up; most individual questions resolved in well under two seconds. This run logged 18 chat calls for Vector RAG, 18 for GraphRAG, and 19 for Hybrid RAG.

---

## 6. Hybrid retrieval in action

Ties back to [§1.5](#15-vectorgraph-fusion--a-different-hybrid) — the same RRF formula, on the actual Q1 output.

**Question:** "Were Scott Derrickson and Ed Wood of the same nationality?" Vector search's top 5, by rank: `Ed Wood(1)`, `Scott Derrickson(2)`, `Ed Wood (film)(3)`, `Adam Collis(4)`, `Sinister (film)(5)`. Graph retrieval separately returned 12 chunks, ranked by how many focus entities point to each one — `Scott Derrickson` came out on top of that list because more of the matched/neighbor entities referenced it, ahead of several hub-leaked chunks pulled in via the "Los Angeles" node ([§3](#3-bugs-found-with-real-examples)).

`score = 1/(60+rank_vector) + 1/(60+rank_graph)`

| Chunk | Vector rank | Graph rank | RRF score |
|---|---:|---:|---:|
| **Scott Derrickson** | 2 | 1 | **0.0325** |
| Ed Wood | 1 | — | 0.0164 |

> **✅ The payoff**
>
> `Scott Derrickson` outranks `Ed Wood` in the fused result despite Ed Wood being vector search's #1 pick — because Scott Derrickson showed up on *both* lists. No reranker model was involved; `0.0161 + 0.0164` beat `0.0164 + 0`. Meanwhile, of the seven noise chunks graph retrieval alone returned — including the Los-Angeles-leaked `David Beckham Academy` and `Guns N' Roses discography` — none scored high enough on either list to survive fusion into the top 6. Arithmetic did what a filter would otherwise have to do by hand.

---

## 7. Toy vs. production-grade GraphRAG

Every row here is a corner this project deliberately cut to stay buildable in a few sessions.

| Concern | This project (toy) | Production-grade |
|---|---|---|
| Extraction | Single-pass per batch, no re-check | Multi-pass "gleaning" — re-prompt to catch entities missed on the first pass |
| Entity resolution | Exact + fuzzy string match (0.8 threshold) | Embedding similarity + external KB linking (e.g. Wikidata), catching lexically-different but semantically-same names |
| Storage | networkx graph, pickled to disk | A dedicated graph database with incremental updates — no full rebuild per new document |
| Communities | Single-level Louvain, one resolution | Hierarchical clustering (e.g. Leiden) at multiple zoom levels — the prerequisite for real global search |
| Retrieval reach | Static 1-hop expansion, capped at 20, ranked by raw edge weight | Weighted multi-hop traversal with degree-based dampening, so hub nodes stop dominating |
| Retrieval correctness | Reads an entity's entire corpus-wide `chunk_ids` backreference ([§3](#3-bugs-found-with-real-examples)'s bug) | Chunk provenance scoped to the traversal path actually walked, not the entity's global history |
| Infrastructure | One-shot script, JSON files, single API key, a call-budget counter | Incremental indexing, per-stage tracing, multi-provider fallback, rate-limit-aware scheduling |

---

## 8. Glossary

| Term | Definition |
|---|---|
| **entity** | A distinct real-world thing — person, place, organization, or work — represented as a graph node. |
| **relation / edge** | A connection between two entities, extracted from text mentioning both; carries a weight (mention count) and provenance (source chunks). |
| **node degree** | The number of edges touching a node — a rough proxy for centrality. |
| **community** | A densely-connected subgroup of nodes found by clustering (Louvain here), maximizing modularity. |
| **chunk_id** | The identifier of the source passage a piece of information came from. |
| **focus entities** | The set of matched entities plus their expanded neighbors, whose chunks get pulled into context. |
| **matched vs. neighbor entities** | Matched = named directly in the question. Neighbor = one graph-hop away from a matched entity. |
| **bridge node** | An entity on a path connecting two otherwise-separate matched entities — computed, never used in retrieval here. |
| **hub node** | An entity with an unusually large `chunk_ids` backreference because it's mentioned across many unrelated documents — a common noise source. |
| **RRF** | Reciprocal Rank Fusion — combines ranked lists via `score = Σ 1/(k+rank)`, no trained reranker needed. |
| **local search** | GraphRAG query mode starting from named entities, expanding outward. |
| **global search** | GraphRAG query mode starting from community summaries, for broad questions. Not implemented in this project. |
| **EM (Exact Match)** | 1 if the normalized predicted answer exactly equals the normalized gold answer, else 0. |
| **F1** | Harmonic mean of token-level precision and recall between predicted and gold answers. |

---

## 9. Appendix: project file map

Reference only — where each concept above actually lives in code.

| Stage | File |
|---|---|
| Config / constants | `config.py` |
| Corpus build | `data_prep/build_corpus.py` |
| Embeddings (shared) | `common/embeddings.py` |
| Retry / backoff | `common/retry.py` |
| Call budget tracking | `common/call_budget.py` |
| Reasoning/answer parsing | `common/answer_parsing.py` |
| Extraction | `graph_rag/extraction.py` |
| Entity resolution | `graph_rag/entity_resolution.py` |
| Graph construction | `graph_rag/graph_build.py` |
| Community detection | `graph_rag/communities.py` |
| Community summarization | `graph_rag/summarization.py` |
| Entity matching (query-time) | `graph_rag/query_entities.py` |
| Local search / context assembly | `graph_rag/local_search.py` |
| GraphRAG answer + pipeline | `graph_rag/answer.py`, `graph_rag/pipeline.py` |
| Vector index + retrieval | `vector_rag/index.py`, `vector_rag/retrieve.py` |
| Vector answer + pipeline | `vector_rag/answer.py`, `vector_rag/pipeline.py` |
| RRF fusion | `hybrid_rag/fuse.py` |
| Hybrid answer + pipeline | `hybrid_rag/answer.py`, `hybrid_rag/pipeline.py` |
| Eval runner | `eval/run_eval.py`, `scripts/04_run_eval.py` |
| Eval scoring (EM/F1) | `eval/scoring.py` |
| Eval retrieval metrics | `eval/retrieval_metrics.py` |
| Eval report | `eval/report.py` |
| Streamlit UI | `app.py` |

---

*Written to be re-read, not shared as a tutorial — every number above traces back to `data/eval_results.json` or the config it was run with.*
