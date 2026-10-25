from rag.document_loader import extract_text
from rag.chunker import split_text

import faiss
import pickle
import os

from sentence_transformers import SentenceTransformer


# -----------------------------
# Paths
# -----------------------------

RESUME_PATH = "uploads/resumes/Resume.pdf"

VECTOR_DB_PATH = "vector_db"

INDEX_PATH = os.path.join(
    VECTOR_DB_PATH,
    "resume.index"
)

CHUNKS_PATH = os.path.join(
    VECTOR_DB_PATH,
    "resume_chunks.pkl"
)


# -----------------------------
# Load Resume
# -----------------------------

print("Loading resume...")

text = extract_text(RESUME_PATH)

print("Resume loaded!")
print("Characters:", len(text))


# -----------------------------
# Split into chunks
# -----------------------------

chunks = split_text(
    text,
    chunk_size=100,
    overlap=20
)

print("\nNumber of chunks:", len(chunks))


# -----------------------------
# Load Embedding Model
# -----------------------------

print("\nLoading embedding model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded!")


# -----------------------------
# Create Embeddings
# -----------------------------

embeddings = model.encode(
    chunks,
    convert_to_numpy=True
).astype("float32")


print("\nEmbedding shape:")
print(embeddings.shape)


# -----------------------------
# Create FAISS Index
# -----------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(
    dimension
)

index.add(embeddings)


# -----------------------------
# Save
# -----------------------------

os.makedirs(
    VECTOR_DB_PATH,
    exist_ok=True
)

faiss.write_index(
    index,
    INDEX_PATH
)

with open(
    CHUNKS_PATH,
    "wb"
) as file:

    pickle.dump(
        chunks,
        file
    )


print("\n================================")
print("Resume indexed successfully!")
print("================================")

print("FAISS index:", INDEX_PATH)
print("Chunks:", CHUNKS_PATH)
print("Vectors:", index.ntotal)