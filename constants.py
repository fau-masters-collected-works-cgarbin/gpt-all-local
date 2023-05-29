"""Constants used throughout the project."""
from chromadb.config import Settings

DATA_DIR = "data"
STORAGE_DIR = "vector_store"

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
    # Configure Chroma for persistence
    # DuckDB seems to be the lightest alternative
    chroma_db_impl="chromadb.db.duckdb.PersistentDuckDB",
    persist_directory=STORAGE_DIR,
    # Do not send telemetry data (the goal of this project is to do everything locally)
    anonymized_telemetry=False
)
