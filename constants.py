"""Constants used throughout the project."""
from chromadb.config import Settings

DATA_DIR = "data"
STORAGE_DIR = "vector_store"

# Control the chunking
# TODO: investigate chunking strategies
# See, for example, https://www.pinecone.io/learn/chunking-strategies/
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Model name for the HuggingFace embeddings
# See https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2: "It maps sentences & paragraphs to a 384
# dimensional dense vector space and can be used for tasks like clustering or semantic search."
# Note that 384 dimensions is not a lot. It may not produce the best results. However, it's small and fast to run on
# a CPU-only machine.
EMBEDDINGS_MODEL_NAME = "all-MiniLM-L6-v2"

CHROMA_SETTINGS = Settings(
    # Configure Chroma for persistence
    # DuckDB seems to be the lightest alternative
    chroma_db_impl="chromadb.db.duckdb.PersistentDuckDB",
    persist_directory=STORAGE_DIR,
    # Do not send telemetry data (the goal of this project is to do everything locally)
    anonymized_telemetry=False
)
