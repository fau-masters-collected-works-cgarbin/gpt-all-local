"""Ingest documents into the vector database.

This code is heavily based on the ingest.py code from https://github.com/imartinez/privateGPT.

Their code has some nice features:

- A map of loaders, making it easy to add/remove loaders.
- Parallel loading of documents.
- Progress bar.

"""
from pathlib import Path
from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import constants
import logger

log = logger.get_logger()

# Extension to loader mapping
# This concept, of having a map, is based on privateGPT's implementation
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PDFMinerLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}


def _store_exists():
    """Return True if the vector store exists."""
    # Ensure that the storage directory exists
    Path(constants.STORAGE_DIR).mkdir(parents=True, exist_ok=True)
    # Check if the vector store exists by checking if the Chroma files exist
    index = Path(constants.STORAGE_DIR) / "index"
    collections = Path(constants.STORAGE_DIR) / "chroma-collections.parquet"
    embeddings = Path(constants.STORAGE_DIR) / "chroma-embeddings.parquet"
    return index.exists() and collections.exists() and embeddings.exists()


def ingest():
    """Ingest all documents in the data directory into the vector store.

    TODO: verify what happens if the document already exists in the store.
    """
    if _store_exists():
        log.debug("The vector store already exists in '%s'.", constants.STORAGE_DIR)
    else:
        log.info("Creating the vector in '%s'.", constants.STORAGE_DIR)
