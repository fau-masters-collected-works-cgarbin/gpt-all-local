"""Constants used throughout the project."""
from pathlib import Path
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

# LLM model
MODEL_DIR = "models"
MODEL_FILE = "ggml-gpt4all-j-v1.3-groovy.bin"
MODEL = Path(MODEL_DIR) / Path(MODEL_FILE)  # use pathlib to work on Windows and Linux
# Number of similar items (chunks) to retrieve from the store
TARGET_SOURCE_CHUNKS = 4
# Maximum token limit for the LLM model
# From the original privateGPT commit https://github.com/imartinez/privateGPT/commit/ad661933cb3def747793c4b7194e3a42d2ab29a5:
# "Number of tokens in the prompt that are fed into the model at a time. Optimal value differs a lot depending
# on the model (8 works well for GPT4All, and 1024 is better for LlamaCpp)"
MODEL_N_BATCH = 8

CHROMA_SETTINGS = Settings(
    # Configure Chroma for persistence
    persist_directory=STORAGE_DIR,
    # Do not send telemetry data (the goal of this project is to do everything locally)
    anonymized_telemetry=False
)
