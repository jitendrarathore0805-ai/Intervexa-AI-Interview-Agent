import json
import logging
from pathlib import Path

import faiss

from rag.chunker import split_text
from rag.embeddings import create_embeddings

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"


def _resume_paths(resume_token):
    # Tokens are UUID hex strings, not arbitrary filesystem paths.
    if (
        not isinstance(resume_token, str)
        or len(resume_token) != 32
        or any(c not in "0123456789abcdef" for c in resume_token.lower())
    ):
        raise ValueError("Invalid resume token.")

    token = resume_token.lower()
    return (
        VECTOR_DB_DIR / f"{token}.index",
        VECTOR_DB_DIR / f"{token}.json"
    )


def index_resume_text(resume_token, resume_text):
    """Create an independent FAISS index for one uploaded resume."""
    if not resume_text or not resume_text.strip():
        raise ValueError("Resume text is empty.")

    chunks = split_text(
        resume_text[:30000],
        chunk_size=180,
        overlap=35
    )

    if not chunks:
        raise ValueError("Could not create resume chunks.")

    embeddings = create_embeddings(chunks)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    index_path, chunks_path = _resume_paths(resume_token)

    # Write temporary files before replacing the saved files.
    temp_index = index_path.with_suffix(".index.tmp")
    temp_chunks = chunks_path.with_suffix(".json.tmp")

    try:
        faiss.write_index(index, str(temp_index))

        with temp_chunks.open("w", encoding="utf-8") as file:
            json.dump(chunks, file, ensure_ascii=False)

        temp_index.replace(index_path)
        temp_chunks.replace(chunks_path)

    except Exception:
        temp_index.unlink(missing_ok=True)
        temp_chunks.unlink(missing_ok=True)
        logger.exception("Failed to index resume")
        raise

    return {
        "chunk_count": len(chunks),
        "index_path": str(index_path)
    }


def load_resume_index(resume_token):
    index_path, chunks_path = _resume_paths(resume_token)

    if not index_path.is_file() or not chunks_path.is_file():
        return None, []

    index = faiss.read_index(str(index_path))

    with chunks_path.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not isinstance(chunks, list):
        raise ValueError("Saved resume chunks are invalid.")

    return index, chunks


def delete_resume_index(resume_token):
    """Remove one resume's vector index and chunk metadata."""
    index_path, chunks_path = _resume_paths(resume_token)

    index_path.unlink(missing_ok=True)
    chunks_path.unlink(missing_ok=True)