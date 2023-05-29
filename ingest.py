"""Ingest documents into the vector database.

This code is heavily based on the ingest.py code from https://github.com/imartinez/privateGPT.

Their code has some nice features:

- A map of loaders, making it easy to add/remove loaders.
- Parallel loading of documents.
- Progress bar.

"""
from pathlib import Path
import time
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


def _store_exists() -> bool:
    """Return True if the vector store exists."""
    # Ensure that the storage directory exists
    Path(constants.STORAGE_DIR).mkdir(parents=True, exist_ok=True)
    # Check if the vector store exists by checking if the Chroma files exist
    index = Path(constants.STORAGE_DIR) / "index"
    collections = Path(constants.STORAGE_DIR) / "chroma-collections.parquet"
    embeddings = Path(constants.STORAGE_DIR) / "chroma-embeddings.parquet"
    return index.exists() and collections.exists() and embeddings.exists()


def _file_list() -> list[Path]:
    """Return a list of files to ingest."""
    files = []
    for ext in LOADER_MAPPING:
        files.extend(Path(constants.DATA_DIR).rglob(f"*{ext}"))
    return files


def _load_one_file(file: Path) -> Document:
    """Load a file into a document."""
    if file.suffix not in LOADER_MAPPING:
        log.error("No loader found for file '%s' - skipping it", file)
        return None

    loader_class, loader_kwargs = LOADER_MAPPING[file.suffix]
    loader = loader_class(str(file), **loader_kwargs)
    return loader.load()[0]


def _load_all_files(files: list[Path]) -> list[Document]:
    """Load all files into documents."""
    documents = []
    for file in files:
        log.debug("Loading file '%s'", file)
        start = time.time()
        document = _load_one_file(file)
        load_time = time.time() - start
        if document is not None:
            log.info("Loaded file '%s' with size %s in %.2f seconds", file, f"{file.stat().st_size:,}", load_time)
            documents.append(document)
    return documents


def ingest():
    """Ingest all documents in the data directory into the vector store.

    TODO: verify what happens if the document already exists in the store, i.e. what happens if we call "ingest"
    multiple times and some of the files have already been ingested.
    """
    embeddings = HuggingFaceEmbeddings(model_name=constants.EMBEDDINGS_MODEL_NAME)
    if _store_exists():
        log.info("The vector store already exists in '%s' - updating it", constants.STORAGE_DIR)
        db = Chroma(persist_directory=constants.STORAGE_DIR, embedding_function=embeddings,
                    client_settings=constants.CHROMA_SETTINGS)
        # We use only one collection for now
        collection = db.get()
        log.debug("Loaded collection: %s", collection)
    else:
        log.info("Creating a new vector store in '%s'", constants.STORAGE_DIR)
        files = _file_list()
        log.info("Found %d files to ingest", len(files))
        log.info("Loading files")
        documents = _load_all_files(files)
    log.debug("Persisting the vector store")
    #db.persist()
