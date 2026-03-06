"""ChromaDB embedding and retrieval for changelog analysis."""

from __future__ import annotations

import hashlib
from typing import Any

from migratowl.config import active_model, settings
from migratowl.core.llm import get_client, get_embedding, get_llm_semaphore
from migratowl.models.schemas import BreakingChange, ChangelogAnalysis, ChangelogSummary, RAGQueryResult

# Safe sub-chunk size for embedding. RST/technical content tokenizes at ~2 chars/token,
# so 4000 chars ≈ 2000 tokens — safely within nomic-embed-text and text-embedding-3-small
# (both cap at 8192 tokens). Oversized version sections are split into multiple sub-chunks;
# no content is discarded.
EMBED_CHUNK_CHARS = 4_000


def _import_chromadb() -> Any:
    """Lazy import chromadb to avoid import errors in test environments."""
    import chromadb as _chromadb

    return _chromadb


def get_collection(project_path: str = "") -> Any:
    """Get or create the changelogs collection with cosine similarity.

    The collection name is namespaced by both the active embedding model and the
    project path so that different projects and embedding backends never share a
    collection.  OpenAI (1536-dim) and Ollama (768-dim) embeddings also remain
    separated since they are incompatible vector dimensions.
    """
    _chromadb = _import_chromadb()
    client = _chromadb.PersistentClient(path=settings.vectorstore_path)
    model = settings.local_embedding_model if settings.use_local_llm else settings.embedding_model
    safe_model = model.replace("/", "_").replace("-", "_").replace(".", "_")
    project_hash = hashlib.sha256(project_path.encode()).hexdigest()[:8]
    return client.get_or_create_collection(
        f"changelogs_{safe_model}_{project_hash}",
        metadata={"hnsw:space": "cosine"},
    )


def purge_dep_embeddings(dep_name: str, project_path: str = "") -> int:
    """Remove all embeddings for a single dependency. Returns count of deleted docs."""
    collection = get_collection(project_path)
    existing = collection.get(where={"dep_name": dep_name})
    ids = existing["ids"]
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def purge_stale_embeddings(
    active_dep_names: set[str], project_path: str = "",
) -> dict[str, int]:
    """Remove embeddings for deps not in active_dep_names. Returns {dep: count} of purged."""
    collection = get_collection(project_path)
    all_docs = collection.get()
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
    """Embed and upsert changelog chunks into ChromaDB.

    Each chunk: {"version": "2.0.0", "content": "..."}
    """
    collection = get_collection(project_path)

    for chunk in version_chunks:
        content = chunk["content"]
        sub_contents = [content[i : i + EMBED_CHUNK_CHARS] for i in range(0, max(len(content), 1), EMBED_CHUNK_CHARS)]
        for idx, sub_content in enumerate(sub_contents):
            embedding = await get_embedding(sub_content)
            doc_id = f"{dep_name}:{chunk['version']}:{idx}"
            collection.upsert(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[sub_content],
                metadatas=[{"dep_name": dep_name, "version": chunk["version"]}],
            )


async def _summarize_changelog(text: str, dep_name: str) -> str:
    """Summarize oversized changelog text before sending to the analysis LLM.

    Called when combined chunk text exceeds SUMMARIZE_THRESHOLD_CHARS.  Returns
    a concise plain-text summary focused on breaking changes and deprecations.
    """
    instructor_client = get_client()
    async with get_llm_semaphore():
        result: ChangelogSummary = await instructor_client.chat.completions.create(
            model=active_model(),
            response_model=ChangelogSummary,
            max_retries=settings.max_retries,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a dependency migration expert. "
                        "Summarize the following changelog text, keeping only breaking changes, "
                        "removals, renames, and deprecations. Discard new features and minor fixes."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Summarize this {dep_name} changelog:\n\n{text}",
                },
            ],
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

    collection = get_collection(project_path)
    query_embedding = await get_embedding(query_text)

    results = collection.query(
        query_embeddings=[query_embedding],
        where={"dep_name": dep_name},
        n_results=n_results,
    )

    documents = results["documents"][0] if results["documents"] else []

    if not documents:
        return RAGQueryResult(breaking_changes=[], confidence=0.0, source_chunks=[])

    combined_text = "\n\n---\n\n".join(documents)

    if len(combined_text) > settings.summarize_threshold:
        combined_text = await _summarize_changelog(combined_text, dep_name)

    instructor_client = get_client()
    async with get_llm_semaphore():
        analysis: ChangelogAnalysis = await instructor_client.chat.completions.create(
            model=active_model(),
            response_model=ChangelogAnalysis,
            max_retries=settings.max_retries,
            messages=[  # type: ignore[arg-type]
                {
                    "role": "system",
                    "content": "You are a dependency migration expert. Analyze the following changelog excerpts and identify breaking changes, deprecations, and new features.",
                },
                {
                    "role": "user",
                    "content": f"Analyze these changelog excerpts for {dep_name}:\n\n{combined_text}",
                },
            ],
        )

    verified_changes, verification_ratio = verify_breaking_changes(
        analysis.breaking_changes, documents
    )
    adjusted_confidence = analysis.confidence * (1 - 0.5 * (1 - verification_ratio))

    return RAGQueryResult(
        breaking_changes=verified_changes,
        confidence=adjusted_confidence,
        source_chunks=documents,
    )
