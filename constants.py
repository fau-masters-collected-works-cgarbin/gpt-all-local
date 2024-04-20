"""Constants used throughout the project."""

from pathlib import Path

DATA_DIR = "data"
STORAGE_DIR = "vector_store"

# Control the chunking
# TODO: investigate chunking strategies
# See, for example, https://www.pinecone.io/learn/chunking-strategies/
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Model name for the HuggingFace embeddings
# See Hugging Face's leaderboard at https://huggingface.co/spaces/mteb/leaderboard
# Pick a model that matches your CPU/GPU capabilities and the number of dimensions you want
EMBEDDINGS_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# LLM model
MODEL_DIR = "models"
MODEL_FILE = "mistral-7b-openorca.gguf2.Q4_0.gguf"
MODEL = Path(MODEL_DIR) / Path(MODEL_FILE)  # use pathlib to work on Windows and Linux
# Number of similar items (chunks) to retrieve from the store
TARGET_SOURCE_CHUNKS = 4
# Maximum token limit for the LLM model
# From the original privateGPT commit https://github.com/imartinez/privateGPT/commit/ad661933cb3def747793c4b7194e3a42d2ab29a5:
# "Number of tokens in the prompt that are fed into the model at a time. Optimal value differs a lot depending
# on the model (8 works well for GPT4All, and 1024 is better for LlamaCpp)"
MODEL_N_BATCH = 1024

# Options to control parsing
# Write the chunks to a file in the same directory as the original file
# Use to debug the parsing and chunking process
PARSING_WRITE_CHUNKS_TO_FILE = True
PARSING_CHUNKED_FILE_SUFFIX = "-chunked.txt"
# Remove empty lines from the text
PARSING_REMOVE_EMPTY_LINES = True
# Minimum length of a line to be considered for parsing
# Lines with fewer characters are ignored
# Set to zero to disable
PARSING_MINIMUM_LINE_LENGTH = 4
# Join split syllables, e.g. "word-" + "\n" + "word" -> "wordword"
PARSING_JOIN_SPLIT_SYLLABLES = True
