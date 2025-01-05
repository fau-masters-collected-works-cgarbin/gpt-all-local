"""Vector store functions to abstract it from the rest of the code."""

import chromadb
import langchain_chroma
import langchain_huggingface
from chromadb.config import Settings
from langchain.docstore.document import Document

import constants
import logger

CHROMA_SETTINGS = Settings(
    # Configure Chroma for persistence
    persist_directory=constants.STORAGE_DIR,
    # Do not send telemetry data (the goal of this project is to do everything locally)
    anonymized_telemetry=False,
)

_DB = None


def _prepared() -> bool:
    """Check if the store is prepared."""
    return _DB is not None


def _prepare():
    """Prepare the vector store.

    Notes:
      - *Not* thread-safe.
    """
    if _prepared():
        return

    global _DB  # pylint: disable=global-statement

    log = logger.get_logger()
    log.debug("Preparing the vector store")
    log.debug(f"   Loading the embedding model '{constants.EMBEDDINGS_MODEL_NAME}'")
    embeddings = langchain_huggingface.HuggingFaceEmbeddings(model_name=constants.EMBEDDINGS_MODEL_NAME)

    log.debug(f"   Creating the vector store in '{constants.STORAGE_DIR}'")
    client = chromadb.PersistentClient(settings=CHROMA_SETTINGS, path=constants.STORAGE_DIR)
    _DB = langchain_chroma.Chroma(persist_directory=constants.STORAGE_DIR, embedding_function=embeddings, client=client)


def store():
    """Return the vector store.

    Use with caution to not break the abstraction.
    """
    _prepare()
    return _DB


def add_documents(documents: list[Document]) -> None:
    """Add the given documents to the vector store."""
    _prepare()
    _DB.add_documents(documents)


def files_in_store() -> list[str]:
    """Return the list of files in the store.

    Only files names are returned, not the full path.
    """
    _prepare()

    # We need to be careful here because we are making assumptions about the format of external data
    try:
        metadata = _DB.get()["metadatas"]
        # Remove duplicates
        set_of_files = set()
        for file_name in metadata:
            set_of_files.add(file_name["source"].split("/")[-1])
        return list(set_of_files)
    except:  # noqa: E722
        return []


def file_stored(file_name: str) -> bool:
    """Check if the file is in the store.

    Note that it's a simple file name comparison, which has these limitations:
      - It doesn't check if the file is in the store with a different name.
      - It doesn't check if the file contents are the same.
    """
    _prepare()

    # We compare only the file name, not the full path
    file_name = file_name.split("/")[-1]
    files_already_in_store = files_in_store()
    return file_name in files_already_in_store


# Use this to debug the code
# Modify the code and start under the debugger
if __name__ == "__main__":
    files = files_in_store()
    print(f"Files in store: {files}")
