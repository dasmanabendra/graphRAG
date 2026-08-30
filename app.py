import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

import config
from common.call_budget import tracker
from data_prep.build_corpus import load_cached_corpus
from data_prep.load_hotpotqa import load_cached_examples
from eval.scoring import exact_match_score, f1_score
from graph_rag.communities import community_members, load_communities
from graph_rag.pipeline import answer_question as graph_answer_question
from graph_rag.pipeline import get_graph
from graph_rag.summarization import load_summaries
from hybrid_rag.pipeline import answer_question as hybrid_answer_question
from vector_rag.index import get_collection
from vector_rag.pipeline import answer_question as vector_answer_question

st.set_page_config(page_title="GraphRAG vs Vector RAG", layout="wide")


@st.cache_data
def _load_examples():
    try:
        return load_cached_examples()
    except FileNotFoundError:
        return []


@st.cache_data
def _load_corpus():
    try:
        return load_cached_corpus()
    except FileNotFoundError:
        return {}


@st.cache_data
def _load_communities_and_summaries():
    try:
        return load_communities(), load_summaries()
    except FileNotFoundError:
        return {}, {}


@st.cache_data
def _chunk_to_question_numbers() -> dict[str, list[int]]:
    """chunk_id (== Wikipedia title) -> 1-based question numbers whose raw
    context included that paragraph. Corpus paragraphs are deduped across
    all questions, so a retrieved chunk on its own doesn't say which
    question(s) it originally came from -- this maps it back so you can
    tell whether a retrieval actually pulled a chunk that belongs to the
    question being asked, or one that only belongs to some other question.
    """
    mapping: dict[str, list[int]] = {}
    for i, example in enumerate(_load_examples()):
        for title in example.get("context", {}).get("title", []):
            mapping.setdefault(title, []).append(i + 1)
    return mapping


def _pipeline_call_delta(before: dict, after: dict, prefix: str) -> int:
    keys = set(before) | set(after)
    return sum(after.get(k, 0) - before.get(k, 0) for k in keys if k.startswith(prefix))


def _question_number_label(chunk_or_title: str, q_map: dict[str, list[int]], current_qnum: Optional[int]) -> str:
    qnums = q_map.get(chunk_or_title, [])
    if not qnums:
        return "not in any question's context"
    label = ", ".join(f"Q#{n}" for n in qnums)
    return label + (" ✅ matches this question" if current_qnum in qnums else "")


def _render_step_detail(detail: dict, current_qnum: Optional[int] = None) -> None:
    for key, value in detail.items():
        label = key.replace("_", " ").capitalize()
        if key == "prompt":
            st.caption("Prompt sent to the model:")
            st.code(value, language=None, wrap_lines=True)
        elif key == "answer":
            st.caption("Raw answer:")
            st.write(value)
        elif key == "results":
            q_map = _chunk_to_question_numbers()
            if current_qnum is not None and value:
                matches = sum(1 for r in value if current_qnum in q_map.get(r["title"], []))
                st.markdown(f"**{len(value)} chunks retrieved -- {matches} from this question, {len(value) - matches} from elsewhere**")
            for r in value:
                q_label = _question_number_label(r["title"], q_map, current_qnum)
                st.write(f"- **{r['title']}** (`{r['chunk_id']}`) -- distance {r['distance']:.3f} -- {q_label}")
            if not value:
                st.write("(no results)")
        elif key == "relations":
            if value:
                for r in value:
                    st.write(f"- {r['from']} --[{r['relation']}]--> {r['to']}")
            else:
                st.write("(no relations from matched entities)")
        elif key == "bridge_nodes":
            st.write("**Bridge nodes:** " + (", ".join(value) if value else "(none)"))
            st.caption(
                "Display only -- never affects retrieval. Only fires when two *matched* entities "
                "share a neighbor they aren't directly connected to; most questions name just one "
                "entity, so this stays empty even when the neighbor expansion above is doing real work."
            )
        elif key == "chunk_sources":
            q_map = _chunk_to_question_numbers()
            graph = get_graph()
            if current_qnum is not None and value:
                matches = sum(1 for cid in value if current_qnum in q_map.get(cid, []))
                st.markdown(f"**{len(value)} chunks pulled in -- {matches} from this question, {len(value) - matches} from elsewhere**")
            if value:
                st.caption(
                    "⚠ marks an entity with zero graph edges -- it reached this chunk purely because "
                    "that entity string was extracted from it during indexing, not because of any graph relation."
                )
            for cid, entities in value.items():
                q_label = _question_number_label(cid, q_map, current_qnum)
                tagged = [f"{e} ⚠" if graph.degree(e) == 0 else e for e in entities if e in graph]
                st.write(f"- `{cid}` -- via {', '.join(tagged)} -- {q_label}")
            if not value:
                st.write("(no source passages)")
        elif key == "fused_results":
            q_map = _chunk_to_question_numbers()
            if current_qnum is not None and value:
                matches = sum(1 for f in value if current_qnum in q_map.get(f["chunk_id"], []))
                st.markdown(f"**{len(value)} chunks after fusion -- {matches} from this question, {len(value) - matches} from elsewhere**")
            for f in value:
                q_label = _question_number_label(f["chunk_id"], q_map, current_qnum)
                via = " + ".join(f["found_via"])
                st.write(f"- `{f['chunk_id']}` -- score {f['score']:.3f} -- found via {via} -- {q_label}")
            if not value:
                st.write("(no results)")
        elif key == "community_summaries":
            if value:
                for s in value:
                    st.markdown(f"> {s}")
            else:
                st.write("(no community summary attached)")
        elif isinstance(value, list):
            st.write(f"**{label}:** " + (", ".join(str(v) for v in value) if value else "(none)"))
        else:
            st.write(f"**{label}:** {value}")


def render_steps(steps: list[dict], current_qnum: Optional[int] = None) -> None:
    for i, step in enumerate(steps, start=1):
        badge = "⚡" if "call" in step["cost"] else "\U0001f193"
        with st.expander(f"{i}. {step['name']}  ·  {badge} {step['cost']}"):
            if step.get("description"):
                st.caption(step["description"])
            _render_step_detail(step["detail"], current_qnum)


def render_ask_a_question_tab() -> None:
    examples = _load_examples()

    with st.expander("Corpus & call budget", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            st.write(f"**Corpus:** {len(examples)} HotpotQA questions indexed")
            try:
                graph = get_graph()
                st.write(f"**Graph:** {graph.number_of_nodes()} entities, {graph.number_of_edges()} relations")
            except FileNotFoundError:
                st.write("Graph not built yet -- run scripts/03_build_graph_index.py")
        with col_b:
            st.write(f"**Call budget today** (cap {config.DAILY_CALL_CAP}/bucket):")
            totals = tracker.totals()
            if totals:
                for key, count in sorted(totals.items()):
                    st.text(f"{key}: {count}")
            else:
                st.text("No calls yet today")

    st.subheader("Ask a question")
    st.markdown(
        "**Vector RAG answers in 3 steps:** embed the question -> similarity search over chunk "
        "embeddings -> generate an answer from the top-k passages.\n\n"
        "**GraphRAG answers in 4 steps:** match entities named in the question -> expand 1-hop "
        "graph neighbors -> gather each focus entity's source passages and community summary -> "
        "generate an answer from that assembled context.\n\n"
        "**Hybrid RAG answers in 4 steps:** run Vector RAG's retrieval and GraphRAG's retrieval "
        "independently -> fuse the two ranked chunk lists with Reciprocal Rank Fusion (a chunk found "
        "by both outranks one found by only one) -> generate an answer from the fused context. Same "
        "1 chat call as the other two; fusion itself is free.\n\n"
        "Ask a question below, then expand any step in the trace to see exactly what happened and why."
    )

    mode = st.radio("Question source", ["Pick from corpus", "Custom question"], horizontal=True)

    question: str = ""
    gold_answer = None
    question_number: Optional[int] = None
    question_id: Optional[str] = None

    if mode == "Pick from corpus" and examples:
        idx = st.selectbox(
            "Question",
            range(len(examples)),
            format_func=lambda i: f"#{i + 1} -- {examples[i]['question'][:80]}",
        )
        question = examples[idx]["question"]
        gold_answer = examples[idx]["answer"]
        question_number = idx + 1
        question_id = examples[idx].get("id")
        st.caption(f"Question #{question_number} of {len(examples)} -- id `{question_id}`")
    elif mode == "Pick from corpus":
        st.warning("No cached corpus found -- run scripts/01_fetch_data.py first, or switch to a custom question.")
    else:
        question = st.text_input("Type a question", value="")
        st.caption("Note: pipelines only know what's in the indexed corpus -- an out-of-corpus question is a good way to see how each one fails.")

    ask = st.button("Ask all three pipelines", type="primary", disabled=not question)

    if ask and question:
        totals_before = tracker.totals()

        with st.spinner("Running vector RAG..."):
            vector_result = vector_answer_question(question)
        totals_after_vector = tracker.totals()
        vector_calls = _pipeline_call_delta(totals_before, totals_after_vector, "vector_rag")

        with st.spinner("Running GraphRAG..."):
            graph_result = graph_answer_question(question)
        totals_after_graph = tracker.totals()
        graph_calls = _pipeline_call_delta(totals_after_vector, totals_after_graph, "graph_rag")

        with st.spinner("Running hybrid RAG..."):
            hybrid_result = hybrid_answer_question(question)
        totals_after_hybrid = tracker.totals()
        hybrid_calls = _pipeline_call_delta(totals_after_graph, totals_after_hybrid, "hybrid_rag")

        if gold_answer:
            q_part = f"Question #{question_number} (id `{question_id}`) -- " if question_number else ""
            st.info(f"{q_part}Gold answer: **{gold_answer}**")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### Vector RAG")
            st.write(vector_result["prediction"])
            if vector_result.get("reasoning"):
                st.caption(f"Reasoning: {vector_result['reasoning']}")
            if gold_answer:
                em = exact_match_score(vector_result["prediction"], gold_answer)
                f1 = f1_score(vector_result["prediction"], gold_answer)
                st.caption(f"EM: {em} | F1: {f1:.2f}")
            st.caption(f"LLM calls this query: {vector_calls}")
            st.markdown("**How it got this answer:**")
            render_steps(vector_result["steps"], question_number)

        with col2:
            st.markdown("### GraphRAG")
            st.write(graph_result["prediction"])
            if graph_result.get("reasoning"):
                st.caption(f"Reasoning: {graph_result['reasoning']}")
            if gold_answer:
                em = exact_match_score(graph_result["prediction"], gold_answer)
                f1 = f1_score(graph_result["prediction"], gold_answer)
                st.caption(f"EM: {em} | F1: {f1:.2f}")
            st.caption(f"LLM calls this query: {graph_calls}")
            st.markdown("**How it got this answer:**")
            render_steps(graph_result["steps"], question_number)

        with col3:
            st.markdown("### Hybrid RAG")
            st.write(hybrid_result["prediction"])
            if hybrid_result.get("reasoning"):
                st.caption(f"Reasoning: {hybrid_result['reasoning']}")
            if gold_answer:
                em = exact_match_score(hybrid_result["prediction"], gold_answer)
                f1 = f1_score(hybrid_result["prediction"], gold_answer)
                st.caption(f"EM: {em} | F1: {f1:.2f}")
            st.caption(f"LLM calls this query: {hybrid_calls}")
            st.markdown("**How it got this answer:**")
            render_steps(hybrid_result["steps"], question_number)
    elif ask and not question:
        st.warning("Enter or select a question first.")


def render_raw_database_tab() -> None:
    examples = _load_examples()
    st.write(
        f"**{len(examples)} raw HotpotQA examples** fetched from `hotpotqa/hotpot_qa` "
        "(distractor split, streamed and cached to `data/raw_hotpotqa.json`)."
    )
    if not examples:
        st.warning("No cached raw examples -- run scripts/01_fetch_data.py first.")
        return

    idx = st.selectbox(
        "Inspect a raw example",
        range(len(examples)),
        format_func=lambda i: examples[i]["question"][:90],
        key="raw_example_picker",
    )
    example = examples[idx]
    st.json(
        {
            "id": example.get("id"),
            "question": example["question"],
            "answer": example["answer"],
            "type": example.get("type"),
            "level": example.get("level"),
            "supporting_facts": example.get("supporting_facts"),
            "context": example.get("context"),
        }
    )

    st.divider()

    corpus = _load_corpus()
    st.write(
        f"**Deduped corpus:** {len(examples)} examples -> {len(corpus)} unique chunks "
        "(deduped by Wikipedia title -- `data/corpus.json`)."
    )
    if corpus:
        rows = [
            {
                "chunk_id": c["chunk_id"],
                "title": c["title"],
                "chars": len(c["text"]),
                "preview": c["text"][:120],
            }
            for c in corpus.values()
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.warning("No cached corpus found -- run scripts/01_fetch_data.py first.")


def render_vector_rag_tab() -> None:
    st.markdown(
        "**Chunking:** none beyond the corpus dedup step -- each Wikipedia paragraph "
        "(already the corpus's unit) is used as-is as the retrieval chunk; no further splitting."
    )

    stage_counts = tracker.stage_counts()
    embed_calls = stage_counts.get("vector_rag/index", 0)
    st.markdown(
        f"**Embedding:** model `{config.EMBEDDING_MODEL}`, task_type `RETRIEVAL_DOCUMENT`, "
        f"{config.EMBEDDING_DIM} dimensions, batch size 16 -- {embed_calls} embedding call(s) today "
        "(cached embeddings from prior runs don't re-call the API, so this is often 0)."
    )

    st.divider()

    try:
        collection = get_collection()
    except Exception as e:
        st.warning(f"Chroma collection not available -- run scripts/02_build_vector_index.py first. ({e})")
        return

    count = collection.count()
    st.write(f"**Stored vectors:** {count} in Chroma collection `corpus` at `{config.CHROMA_DIR}`.")
    if not count:
        st.warning("Collection is empty -- run scripts/02_build_vector_index.py first.")
        return

    data = collection.get(include=["embeddings", "metadatas", "documents"])
    rows = []
    for cid, meta, emb, doc in zip(data["ids"], data["metadatas"], data["embeddings"], data["documents"]):
        preview = ", ".join(f"{v:.3f}" for v in list(emb)[:6])
        rows.append(
            {
                "chunk_id": cid,
                "title": meta.get("title", ""),
                "text": doc,
                "embedding_preview": f"[{preview}, ...]",
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def render_graph_rag_tab() -> None:
    st.markdown("**GraphRAG indexing runs in 5 steps:**")
    st.markdown(
        "1. **Extraction** (LLM) -- read each chunk, pull out entities (people, orgs, locations, "
        "works, events) and the relations stated between them *within that chunk's text*.\n"
        "2. **Entity resolution** (free, local) -- merge duplicate names for the same entity "
        "(e.g. \"Ed Wood\" / \"Edward Davis Wood Jr.\") via exact + fuzzy string matching. No LLM call.\n"
        "3. **Graph construction** (free, local) -- entities become nodes, relations become edges. "
        "Edges only ever connect entities named together in the same chunk -- but a shared node "
        "(the same entity mentioned in several chunks) indirectly links those chunks together.\n"
        "4. **Community detection** (free, local -- Louvain algorithm) -- group densely-connected "
        "nodes into clusters.\n"
        "5. **Community summarization** (LLM) -- write a short summary for each cluster with 3+ "
        "members; smaller clusters get a free templated stub instead of an LLM call."
    )
    st.divider()

    try:
        graph = get_graph()
    except FileNotFoundError:
        st.warning("Graph not built yet -- run scripts/03_build_graph_index.py first.")
        return

    assignment, summaries = _load_communities_and_summaries()
    members = community_members(assignment) if assignment else {}
    stage_counts = tracker.stage_counts()

    relation_count = sum(len(data.get("provenance", [])) for _, _, data in graph.edges(data=True))
    st.markdown(
        f"**Step 1 -- Extraction:** {graph.number_of_nodes()} resolved entities, {relation_count} relation mentions "
        f"across {graph.number_of_edges()} unique edges -- "
        f"{stage_counts.get('graph_rag/extraction', 0)} extraction call(s) today "
        f"(LLM extracts entities/relations per batch of {config.EXTRACTION_BATCH_SIZE} chunks)."
    )

    st.divider()

    st.markdown(
        "**Step 2 -- Entity resolution** (free, local -- exact + type-scoped fuzzy match on surface forms). "
        "Entities with merged aliases surface first:"
    )
    resolution_rows = [
        {
            "entity": node,
            "type": data.get("type", ""),
            "aliases": len(data.get("aliases", [])),
            "alias_list": ", ".join(data.get("aliases", [])),
        }
        for node, data in graph.nodes(data=True)
    ]
    resolution_rows.sort(key=lambda r: -r["aliases"])
    st.dataframe(resolution_rows, use_container_width=True, hide_index=True)

    st.divider()

    st.markdown(
        f"**Step 3 -- Graph construction** (free, local): {graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges. Top entities by degree (hub nodes -- "
        "disproportionately drive local search):"
    )
    hub_rows = [
        {"entity": node, "type": graph.nodes[node].get("type", ""), "degree": degree}
        for node, degree in sorted(graph.degree, key=lambda x: -x[1])[:15]
    ]
    st.dataframe(hub_rows, use_container_width=True, hide_index=True)

    st.divider()

    st.markdown(f"**Step 4 -- Community detection** (free, local -- Louvain): {len(members)} communities.")
    if members:
        community_rows = [
            {
                "community_id": cid,
                "members": len(nodes),
                "entities": ", ".join(nodes[:8]) + (", ..." if len(nodes) > 8 else ""),
            }
            for cid, nodes in sorted(members.items())
        ]
        st.dataframe(community_rows, use_container_width=True, hide_index=True)
    else:
        st.warning("No communities found -- run scripts/03_build_graph_index.py first.")
        return

    st.divider()

    st.markdown(
        f"**Step 5 -- Community summarization:** {stage_counts.get('graph_rag/summarization', 0)} summarization "
        f"call(s) today -- communities with fewer than {config.MIN_COMMUNITY_SIZE_FOR_SUMMARY} members "
        "get a free templated stub instead of an LLM call."
    )
    stub_count = sum(1 for nodes in members.values() if len(nodes) < config.MIN_COMMUNITY_SIZE_FOR_SUMMARY)
    llm_count = len(members) - stub_count
    st.write(f"{llm_count} communities got a real LLM summary; {stub_count} got the free template.")

    summary_ids = sorted(members)
    picked_cid = st.selectbox(
        "Inspect a community",
        summary_ids,
        format_func=lambda cid: (
            f"Community {cid} ({len(members[cid])} members) -- "
            + ("template" if len(members[cid]) < config.MIN_COMMUNITY_SIZE_FOR_SUMMARY else "LLM summary")
        ),
        key="community_picker",
    )
    picked_nodes = members[picked_cid]
    st.write(f"**Members:** {', '.join(picked_nodes)}")
    st.write(f"**Summary:** {summaries.get(picked_cid, '(no summary)')}")


st.title("GraphRAG vs Vector RAG")
st.caption(
    "Same corpus (a small HotpotQA subset), same question, three retrieval strategies -- "
    "compare answers, retrieved context, and LLM call cost side by side."
)

tab_ask, tab_db = st.tabs(["Ask a question", "Database explorer"])

with tab_ask:
    render_ask_a_question_tab()

with tab_db:
    st.caption("How the raw data became each pipeline's index -- no questions asked yet, this is all indexing-time.")
    sub_raw, sub_vector, sub_graph = st.tabs(["Raw Database", "Vector RAG Indexing", "GraphRAG Indexing"])
    with sub_raw:
        render_raw_database_tab()
    with sub_vector:
        render_vector_rag_tab()
    with sub_graph:
        render_graph_rag_tab()
