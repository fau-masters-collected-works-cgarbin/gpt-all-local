"""Constants used throughout the project."""
from chromadb.config import Settings

DATA_DIR = "data"

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
    # Configure Chroma for persistence
    # DuckDB seems to be the lightest alternative
    chroma_db_impl="chromadb.db.duckdb.PersistentDuckDB",
    persist_directory="vector_store",
    # Do not send telemetry data (the goal of this project is to do everything locally)
    anonymized_telemetry=False
)
