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


def _file_list() -> list[Path]:
    """Return a list of files to ingest."""
    files = []
    for ext in LOADER_MAPPING:
        files.extend(Path(constants.DATA_DIR).rglob(f"*{ext}"))
    return files


def _load_document(file: Path) -> Document:
    """Load a file into a document."""
    if file.suffix not in LOADER_MAPPING:
        log.error("No loader found for file '%s' - skipping it", file)
        return None

    loader_class, loader_kwargs = LOADER_MAPPING[file.suffix]
    start_time = time.time()
    loader = loader_class(str(file), **loader_kwargs)
    # TODO: defer loading (lazy load) until the document is actually needed (when we split it)
    document = loader.load()[0]  # loader is a generator - this forces it to read the file
    elapsed_time = time.time() - start_time
    log.info("   Loaded document of size %s in %.2f seconds", f"{len(document.page_content):,}", elapsed_time)
    return document


def _split_document(document: Document) -> list[Document]:
    """Split a document into chunks."""
    start_time = time.time()
    splitter = RecursiveCharacterTextSplitter(chunk_size=constants.CHUNK_SIZE, chunk_overlap=constants.CHUNK_OVERLAP)
    split_doc = splitter.split_documents(document)
    elapsed_time = time.time() - start_time
    log.info("   Split into %d chunks in %.2f seconds", len(split_doc), elapsed_time)
    return split_doc


def _add_to_store(documents: list[Document], store: any) -> None:
    """Add documents to the vector store.

    Adding to the store also create the embeddings.
    """
    start_time = time.time()
    store.add_documents(documents)
    elapsed_time = time.time() - start_time
    log.info("   Embedded to the vector store in %.2f seconds", elapsed_time)


def _load_all_files(files: list[Path]) -> None:
    """Load all files into documents."""
    embeddings = HuggingFaceEmbeddings(model_name=constants.EMBEDDINGS_MODEL_NAME)
    # TODO: investigate how to correctly update the store when processing documents that already exist in the store
    db = Chroma(persist_directory=constants.STORAGE_DIR, embedding_function=embeddings,
                client_settings=constants.CHROMA_SETTINGS)

    # TODO: Parallelize this loop (load, split, add to store in parallel for each file)
    for file in files:
        log.info("Processing file '%s', with file size %s", file, f"{file.stat().st_size:,}")
        document = _load_document(file)
        if document is not None:
            chunks = _split_document([document])
            _add_to_store(chunks, db)

    # Save once at the end to avoid saving multiple times
    # TODO: investigate if we can save one document at a time, to cover the case where the process is interrupted and
    # we lose all the work, and to save memory (not have all documents in memory at the same time)
    start_time = time.time()
    db.persist()
    elapsed_time = time.time() - start_time
    log.info("Persisted the vector store in %.2f seconds", elapsed_time)


def ingest():
    """Ingest all documents in the data directory into the vector store.

    TODO: verify what happens if the document already exists in the store, i.e. what happens if we call "ingest"
    multiple times and some of the files have already been ingested.
    """
    # Ensure that the storage directory exists
    Path(constants.STORAGE_DIR).mkdir(parents=True, exist_ok=True)

    files = _file_list()
    log.info("Found %d files to ingest", len(files))
    log.info("Loading files")
    _load_all_files(files)
