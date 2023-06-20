"""Ingest documents into the vector database (store).

The goal of this step is to prepare the local data to be used by the language model (LLM). This is done by:

1. Loading the documents from the data directory.
2. Splitting the documents into chunks (to fit in the LLM context window).
3. Extracting the embeddings for each chunk to use in similarity searches.
4. Persisting the embeddings in the vector database.

This code is heavily based on the ingest.py code from https://github.com/imartinez/privateGPT.
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
from langchain.docstore.document import Document
import constants
import logger
import vector_store

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


def _file_list(directory: str) -> list[Path]:
    """Return a list of files to ingest."""
    files = []
    for ext in LOADER_MAPPING:
        files.extend(Path(directory).rglob(f"*{ext}"))
    return files


def _load_document(file: Path) -> Document | None:
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

    # TODO: improve PDF loader
    # This is a kludge to improve what we get from the PDF loader. There are several PDF loaders, with different
    # capabilities. The one we are using is the simplest one, and it doesn't do a good job with some PDFs (mainly
    # because we are using all its features -- see https://stackoverflow.com/a/69151177).
    # In the future we should call the PDF loader directly, not through LangChain, to better control how it loads
    # the document.
    # This joins words that were split at the end of a line (e.g. "word-" + "\n" + "word" -> "wordword")
    document.page_content = document.page_content.replace("-\n", "")

    log.debug("   Loaded document with %s characters in %.2f seconds", f"{len(document.page_content):,}", elapsed_time)
    return document


def _split_document(document: Document) -> list[Document]:
    """Split a document into chunks."""
    start_time = time.time()
    splitter = RecursiveCharacterTextSplitter(chunk_size=constants.CHUNK_SIZE, chunk_overlap=constants.CHUNK_OVERLAP)
    split_doc = splitter.split_documents([document])  # convert to list to satisfy the interface
    elapsed_time = time.time() - start_time
    num_chunks = len(split_doc)
    log.debug("   Split into %d chunks in %.2f seconds", num_chunks, elapsed_time)
    log.debug("   Requested chunk size: %d, average chunk size: %.2f", constants.CHUNK_SIZE, len(document.page_content)/num_chunks)
    return split_doc


def _add_to_store(documents: list[Document], store: any) -> None:
    """Add documents to the vector store.

    This function adds the documents as they are to the store. Documents must be already split
    into chunks, if so desired.

    Adding to the store also creates the embeddings.
    """
    start_time = time.time()
    store.add_documents(documents)
    elapsed_time = time.time() - start_time
    log.debug("   Embedded to the vector store in %.2f seconds", elapsed_time)


def _load_all_files(files: list[Path]) -> None:
    """Load all files into documents."""
    db = vector_store.store()

    files_in_store = vector_store.files_in_store(db)
    log.debug("Found these files in the store: %s", files_in_store)

    # TODO: Parallelize this loop (load, split, add to store in parallel for each file)
    processed_files = 0
    for i, file in enumerate(files):
        log.info("Processing file '%s' (%d of %d), with size %s bytes", file, i+1, len(files),
                 f"{file.stat().st_size:,}")

        # TODO: investigate how to correctly update the store when processing documents that already exist in it
        # The file may have changed since the last time we processed it
        if str(file) in files_in_store:
            log.info("   Skipping because it is already in the store")
            continue

        document = _load_document(file)
        if document is not None:
            chunks = _split_document(document)
            _add_to_store(chunks, db)
            processed_files += 1

    # Save once at the end to avoid saving multiple times
    # TODO: investigate if we can save one document at a time, to cover the case where the process is interrupted and
    # we lose all the work, and to save memory (not have all documents in memory at the same time)
    if processed_files > 0:
        start_time = time.time()
        db.persist()
        elapsed_time = time.time() - start_time
        log.info("Persisted the vector store in %.2f seconds", elapsed_time)


def ingest(directory: str = constants.DATA_DIR):
    """Ingest all documents in a directory into the vector store.

    TODO: verify what happens if the document already exists in the store, i.e. what happens if we call "ingest"
    multiple times and some of the files have already been ingested.
    """
    # Ensure that the storage directory exists
    Path(constants.STORAGE_DIR).mkdir(parents=True, exist_ok=True)

    files = _file_list(directory)
    log.info("Found %d files to ingest in %s", len(files), directory)
    _load_all_files(files)


# Use this to debug the code
# Modify the code and start under the debugger
if __name__ == "__main__":
    logger.set_verbose(True)
    ingest()
