"""Ingest documents into the vector database (store).

The goal of this step is to prepare the local data to be used by the language model (LLM). This is done by:

1. Loading the documents from the data directory.
2. Splitting the documents into chunks (to fit in the LLM context window).
3. Extracting the embeddings for each chunk to use in similarity searches.
4. Persisting the embeddings in the vector database.

This code is heavily based on the ingest.py code from https://github.com/imartinez/privateGPT.
"""

import re
import ssl
import time
from concurrent.futures import ThreadPoolExecutor
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
        log.error(f"No loader found for file '{file}' - skipping it")
        return None

    loader_class, loader_kwargs = LOADER_MAPPING[file.suffix]
    start_time = time.time()
    loader = loader_class(str(file), **loader_kwargs)
    # TODO: defer loading (lazy load) until the document is actually needed (when we split it)
    document = loader.load()[0]  # loader is a generator - this forces it to read the file
    elapsed_time = time.time() - start_time

    document = _post_process_document(document)

    log.debug(f"   Loaded document with {len(document.page_content):,} characters in {elapsed_time:.2f} seconds")
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

    log.debug(f"   Split into {num_chunks} chunks in {elapsed_time:.2f} seconds")
    log.debug(
        f"   Requested chunk size: {constants.CHUNK_SIZE}"
        f", minimum: {min_chunk_size}, maximum: {max_chunk_size}, average: {int(average_chunk_size)}"
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
    log.debug(f"   Embedded to the vector store in {elapsed_time:.2f} seconds")


def _load_all_files(files: list[Path]) -> None:
    """Load all files into documents."""
    # TODO: Parallelize this loop (load, split, add to store in parallel for each file)
    processed_files = 0
    for i, file in enumerate(files):
        log.info(f"Processing file '{file}' ({i + 1} of {len(files)}), with size {file.stat().st_size:,} bytes")

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


def _prepare_to_ingest():
    """Prepare the environment for ingestion."""
    # Workaround for the error "CERTIFICATE_VERIFY_FAILED] certificate verify failed" when downloadig nltk files
    # They are downloaded by parser packages via unstructured
    # Source: https://github.com/gunthercox/ChatterBot/issues/930
    ssl._create_default_https_context = ssl._create_unverified_context

    # Ensure that the storage directory exists
    Path(constants.STORAGE_DIR).mkdir(parents=True, exist_ok=True)

    # Lazy import to log its cost at the point of use
    log.info("Preparing NTLK data")
    import nltk
    from nltk.data import find

    # Download NLTK modules used by Unstructured
    def download_nltk_data(module: str):
        try:
            find(module)
            log.info(f"NLTK module already downloaded: {module}")
        except LookupError:
            log.info(f"Downloading NLTK module: {module}")
            nltk.download(module)

    nltk_modules = ["tokenizers/punkt", "taggers/averaged_perceptron_tagger"]
    with ThreadPoolExecutor() as executor:
        executor.map(download_nltk_data, nltk_modules)


def ingest(directory: str = constants.DATA_DIR):
    """Ingest all documents in a directory into the vector store.

    TODO: verify what happens if the document already exists in the store, i.e. what happens if we call "ingest"
    multiple times and some of the files have already been ingested.
    """
    _prepare_to_ingest()

    files = _file_list(directory)

    # Remove from the list chunked files we may have created in a previous run
    files = [file for file in files if not file.name.endswith(constants.PARSING_CHUNKED_FILE_SUFFIX)]

    log.info(f"Found {len(files)} files to ingest in {directory}")
    _load_all_files(files)


# Use this to debug the code
# Modify the code and start under the debugger
if __name__ == "__main__":
    logger.set_verbose(True)
    ingest()
