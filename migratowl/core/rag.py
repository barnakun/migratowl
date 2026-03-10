"""LangChain-Chroma embedding and retrieval for changelog analysis."""

from __future__ import annotations

import hashlib

from langchain_chroma import Chroma

from migratowl.config import settings
from migratowl.core.llm import get_embeddings, get_llm_semaphore, get_structured_llm
from migratowl.core.prompts import CHANGELOG_ANALYSIS_PROMPT, CHANGELOG_SUMMARY_PROMPT
from migratowl.models.schemas import BreakingChange, ChangelogAnalysis, ChangelogSummary, RAGQueryResult

# Safe sub-chunk size for embedding. RST/technical content tokenizes at ~2 chars/token,
# so 4000 chars ≈ 2000 tokens — safely within nomic-embed-text and text-embedding-3-small
# (both cap at 8192 tokens). Oversized version sections are split into multiple sub-chunks;
# no content is discarded.
EMBED_CHUNK_CHARS = 4_000


_vectorstore_cache: dict[str, Chroma] = {}


def get_vectorstore(project_path: str = "") -> Chroma:
    """Get or create the changelogs vector store with cosine similarity.

    The collection name is namespaced by both the active embedding model and the
    project path so that different projects never share a collection.
    Instances are cached per project_path to avoid recreating Chroma objects.
    """
    if project_path in _vectorstore_cache:
        return _vectorstore_cache[project_path]

    model = settings.embedding_model
    safe_model = model.replace("/", "_").replace("-", "_").replace(".", "_").replace(":", "_")
    project_hash = hashlib.sha256(project_path.encode()).hexdigest()[:8]
    vs = Chroma(
        persist_directory=settings.vectorstore_path,
        embedding_function=get_embeddings(),
        collection_name=f"changelogs_{safe_model}_{project_hash}",
        collection_metadata={"hnsw:space": "cosine"},
    )
    _vectorstore_cache[project_path] = vs
    return vs


def reset_vectorstore_cache() -> None:
    """Clear the vectorstore cache. Used for test cleanup."""
    _vectorstore_cache.clear()


def purge_dep_embeddings(dep_name: str, project_path: str = "") -> int:
    """Remove all embeddings for a single dependency. Returns count of deleted docs."""
    vectorstore = get_vectorstore(project_path)
    existing = vectorstore.get(where={"dep_name": dep_name})
    ids = existing["ids"]
    if ids:
        vectorstore.delete(ids=ids)
    return len(ids)


def purge_stale_embeddings(
    active_dep_names: set[str],
    project_path: str = "",
) -> dict[str, int]:
    """Remove embeddings for deps not in active_dep_names. Returns {dep: count} of purged."""
    vectorstore = get_vectorstore(project_path)
    all_docs = vectorstore.get()
    embedded_deps: set[str] = set()
    for meta in all_docs.get("metadatas", []):
        if meta and "dep_name" in meta:
            embedded_deps.add(meta["dep_name"])

    orphaned = embedded_deps - active_dep_names
    purged: dict[str, int] = {}
    for dep in orphaned:
        count = purge_dep_embeddings(dep, project_path)
        if count:
            purged[dep] = count
    return purged


async def embed_changelog(dep_name: str, version_chunks: list[dict], project_path: str = "") -> None:
    """Embed and store changelog chunks into the vector store.

    Each chunk: {"version": "2.0.0", "content": "..."}
    """
    vectorstore = get_vectorstore(project_path)

    # Purge existing embeddings for this dep to avoid duplicate ID errors
    # (aadd_texts calls add(), not upsert(), so re-runs would fail otherwise)
    purge_dep_embeddings(dep_name, project_path)

    all_texts: list[str] = []
    all_metadatas: list[dict[str, str]] = []
    all_ids: list[str] = []

    # Track how many times each version has been seen to generate unique IDs
    # (changelogs can mention the same version in multiple sections)
    version_counts: dict[str, int] = {}

    for chunk in version_chunks:
        content = chunk["content"]
        version = chunk["version"]
        base_idx = version_counts.get(version, 0)
        sub_contents = [content[i : i + EMBED_CHUNK_CHARS] for i in range(0, max(len(content), 1), EMBED_CHUNK_CHARS)]
        for idx, sub_content in enumerate(sub_contents):
            all_texts.append(sub_content)
            all_metadatas.append({"dep_name": dep_name, "version": version})
            all_ids.append(f"{dep_name}:{version}:{base_idx + idx}")
        version_counts[version] = base_idx + len(sub_contents)

    if all_texts:
        await vectorstore.aadd_texts(texts=all_texts, metadatas=all_metadatas, ids=all_ids)


async def _summarize_changelog(text: str, dep_name: str) -> str:
    """Summarize oversized changelog text before sending to the analysis LLM.

    Called when combined chunk text exceeds SUMMARIZE_THRESHOLD_CHARS.  Returns
    a concise plain-text summary focused on breaking changes and deprecations.
    """
    structured_llm = get_structured_llm(ChangelogSummary)
    chain = CHANGELOG_SUMMARY_PROMPT | structured_llm
    async with get_llm_semaphore():
        result: ChangelogSummary = await chain.ainvoke(
            {
                "dep_name": dep_name,
                "text": text,
            }
        )
    return result.summary


def verify_breaking_changes(
    breaking_changes: list[BreakingChange],
    source_chunks: list[str],
) -> tuple[list[BreakingChange], float]:
    """Cross-check LLM-identified breaking changes against source chunks.

    Returns updated list (with verified field set) and verification ratio.
    """
    if not breaking_changes:
        return [], 1.0

    combined = "\n".join(source_chunks).lower()
    verified_count = 0
    result: list[BreakingChange] = []

    for bc in breaking_changes:
        name = bc.api_name.replace("(", "").replace(")", "")
        parts = [p for p in name.split(".") if len(p) >= 3]
        found = any(part.lower() in combined for part in parts)
        result.append(bc.model_copy(update={"verified": found}))
        if found:
            verified_count += 1

    ratio = verified_count / len(breaking_changes)
    return result, ratio


async def query(
    query_text: str,
    dep_name: str,
    n_results: int | None = None,
    project_path: str = "",
) -> RAGQueryResult:
    """Query changelog embeddings and analyze with LLM.

    Returns RAGQueryResult with breaking changes, confidence, and source chunks.

    n_results defaults to settings.max_rag_results at runtime.
    """
    if n_results is None:
        n_results = settings.max_rag_results

    vectorstore = get_vectorstore(project_path)
    docs = await vectorstore.asimilarity_search(query_text, k=n_results, filter={"dep_name": dep_name})

    documents = [doc.page_content for doc in docs]

    if not documents:
        return RAGQueryResult(breaking_changes=[], confidence=0.0, source_chunks=[])

    combined_text = "\n\n---\n\n".join(documents)

    if len(combined_text) > settings.summarize_threshold:
        combined_text = await _summarize_changelog(combined_text, dep_name)

    structured_llm = get_structured_llm(ChangelogAnalysis)
    chain = CHANGELOG_ANALYSIS_PROMPT | structured_llm
    async with get_llm_semaphore():
        analysis: ChangelogAnalysis = await chain.ainvoke(
            {
                "dep_name": dep_name,
                "combined_text": combined_text,
            }
        )

    verified_changes, verification_ratio = verify_breaking_changes(analysis.breaking_changes, documents)
    adjusted_confidence = analysis.confidence * (1 - 0.5 * (1 - verification_ratio))

    return RAGQueryResult(
        breaking_changes=verified_changes,
        confidence=adjusted_confidence,
        source_chunks=documents,
    )
