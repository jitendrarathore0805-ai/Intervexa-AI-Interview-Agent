import logging

from rag.embeddings import embed_query
from rag.vector_store import load_resume_index

logger = logging.getLogger(__name__)


def retrieve_resume_chunks(query, resume_token, top_k=4):
    """Retrieve relevant chunks only from the specified resume."""
    if not query or not query.strip() or not resume_token:
        return []

    index, chunks = load_resume_index(resume_token)

    if index is None or not chunks:
        return []

    query_embedding = embed_query(query)

    if query_embedding is None:
        return []

    count = min(max(1, int(top_k)), len(chunks))
    scores, indices = index.search(query_embedding, count)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue

        results.append({
            "text": chunks[idx],
            "similarity": float(score)
        })

    return results


def retrieve_context(query, top_k=3, resume_token=None):
    """
    Return context from the specified uploaded resume.

    No global sample-resume index is used.
    """
    if not resume_token:
        return ""

    try:
        results = retrieve_resume_chunks(
            query=query,
            resume_token=resume_token,
            top_k=top_k
        )

        return "\n\n".join(
            f"[Resume excerpt {i}]\n{item['text']}"
            for i, item in enumerate(results, start=1)
        )

    except Exception:
        logger.exception("Resume-specific retrieval failed")
        return ""