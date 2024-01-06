"""Ingest documents into the vector database (store).

The goal of this step is to prepare the local data to be used by the language model (LLM). This is done by:

1. Loading the documents from the data directory.
2. Splitting the documents into chunks (to fit in the LLM context window).
3. Extracting the embeddings for each chunk to use in similarity searches.
4. Persisting the embeddings in the vector database.

This code is heavily based on the ingest.py code from https://github.com/imartinez/privateGPT.
"""
import re
import time
from pathlib import Path

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    CSVLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPDFLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

import constants
import logger
import vector_store

log = logger.get_logger()

# Extension to loader mapping
# This map concept is based on privateGPT's implementation
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (UnstructuredPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}


def _file_list(directory: str) -> list[Path]:
    """Return a list of files to ingest."""
    files = []
    for ext in LOADER_MAPPING:
        files.extend(Path(directory).rglob(f"*{ext}"))
    return files


def _post_process_document(document: Document) -> Document:
    """Post-process a document after loading it.

    These are simple heuristics to improve what we get from the loaders.

    There are several PDF loaders, with different capabilities. In the future we should call the loaders directly,
    not through LangChain, to better control how it loads the document.
    """
    if constants.PARSING_REMOVE_EMPTY_LINES:
        document.page_content = "\n".join([line for line in document.page_content.split("\n") if line.strip() != ""])
    # Note that we have to apply the syllable heuristic removing empty lines to properly detect the split syllables
    # with the simple replacement below
    if constants.PARSING_JOIN_SPLIT_SYLLABLES:
        # Syllables split across lines
        document.page_content = document.page_content.replace("-\n", "")
        # Syllables split within one piece of text
        # Not sure why we get these - it seems PDF loaders understand the text as a single line with a soft line break
        # Use a regular expression that checks for a few characters, followed by a dash, followed by a few other
        # characters to prevent false positives
        document.page_content = re.sub(r"(\w\w)-\s(\w\w)", r"\1\2", document.page_content)

    # Remove lines that are too short afer applying the other heuristics that change the length of the lines
    if constants.PARSING_MINIMUM_LINE_LENGTH > 0:
        document.page_content = "\n".join(
            [
                line
                for line in document.page_content.split("\n")
                if len(line.strip()) >= constants.PARSING_MINIMUM_LINE_LENGTH
            ]
        )
    return document


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

    document = _post_process_document(document)

    log.debug("   Loaded document with %s characters in %.2f seconds", f"{len(document.page_content):,}", elapsed_time)
    return document


def _split_document(document: Document) -> list[Document]:
    """Split a document into chunks."""
    start_time = time.time()
    # TODO: choose a splitter based on the document type
    splitter = RecursiveCharacterTextSplitter(chunk_size=constants.CHUNK_SIZE, chunk_overlap=constants.CHUNK_OVERLAP)
    split_doc = splitter.split_documents([document])  # convert to list to satisfy the interface
    elapsed_time = time.time() - start_time

    # Calculate some statistics
    num_chunks = len(split_doc)
    chunk_sizes = [len(doc.page_content) for doc in split_doc]
    average_chunk_size = sum(chunk_sizes) / num_chunks
    min_chunk_size = min(chunk_sizes)
    max_chunk_size = max(chunk_sizes)

    log.debug("   Split into %d chunks in %.2f seconds", num_chunks, elapsed_time)
    log.debug(
        "   Requested chunk size: %d, minimum, maximum, average chunk size: %d, %d, %.2f",
        constants.CHUNK_SIZE,
        min_chunk_size,
        max_chunk_size,
        average_chunk_size,
    )
    return split_doc


def _add_to_store(documents: list[Document]) -> None:
    """Add documents to the vector store.

    This function adds the documents as they are to the store. Documents must be already split
    into chunks, if so desired.

    Adding to the store also creates the embeddings.
    """
    start_time = time.time()
    vector_store.add_documents(documents)
    elapsed_time = time.time() - start_time
    log.debug("   Embedded to the vector store in %.2f seconds", elapsed_time)


def _load_all_files(files: list[Path]) -> None:
    """Load all files into documents."""
    # TODO: Parallelize this loop (load, split, add to store in parallel for each file)
    processed_files = 0
    for i, file in enumerate(files):
        log.info(
            "Processing file '%s' (%d of %d), with size %s bytes", file, i + 1, len(files), f"{file.stat().st_size:,}"
        )

        # TODO: investigate how to correctly update the store when processing documents that already exist in it
        # The file may have changed since the last time we processed it
        if vector_store.file_stored(str(file)):
            log.info("   Skipping because it is already in the store")
            continue

        document = _load_document(file)
        if document is not None:
            chunks = _split_document(document)

            if constants.PARSING_WRITE_CHUNKS_TO_FILE:
                chunked_file = file.with_name(file.stem + constants.PARSING_CHUNKED_FILE_SUFFIX)
                with chunked_file.open("w") as f:
                    for i, chunk in enumerate(chunks):
                        f.write(f"\n------------------\nChunk {i+1} with length {len(chunk.page_content)}\n\n")
                        f.write(chunk.page_content)
                        f.write("\n")

            _add_to_store(chunks)
            processed_files += 1

    # Save once at the end to avoid saving multiple times
    # TODO: investigate if we can save one document at a time, to cover the case where the process is interrupted and
    # we lose all the work, and to save memory (not have all documents in memory at the same time)
    if processed_files > 0:
        start_time = time.time()
        vector_store.persist()
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

    # Remove from the list chunked files we may have created in a previous run
    files = [file for file in files if not file.name.endswith(constants.PARSING_CHUNKED_FILE_SUFFIX)]

    log.info("Found %d files to ingest in %s", len(files), directory)
    _load_all_files(files)


# Use this to debug the code
# Modify the code and start under the debugger
if __name__ == "__main__":
    logger.set_verbose(True)
    ingest()
