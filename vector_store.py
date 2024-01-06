"""Vector store functions to abstract it from the rest of the code."""
import chromadb
from chromadb.config import Settings
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

import constants

CHROMA_SETTINGS = Settings(
    # Configure Chroma for persistence
    persist_directory=constants.STORAGE_DIR,
    # Do not send telemetry data (the goal of this project is to do everything locally)
    anonymized_telemetry=False,
)

_embeddings = HuggingFaceEmbeddings(model_name=constants.EMBEDDINGS_MODEL_NAME)

_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS, path=constants.STORAGE_DIR)
_db = Chroma(persist_directory=constants.STORAGE_DIR, embedding_function=_embeddings, client=_client)


def store():
    """Return the vector store.

    Use with caution to not break the abstraction.
    """
    return _db


def add_documents(documents: list[Document]) -> None:
    """Add the given documents to the vector store."""
    _db.add_documents(documents)


def persist():
    """Persist the vector store."""
    _db.persist()


def files_in_store() -> list[str]:
    """Return the list of files in the store.

    Only files names are returned, not the full path.
    """
    # We need to be careful here because we are making assumptions about the format of external data
    try:
        metadata = _db.get()["metadatas"]
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
    # We compare only the file name, not the full path
    file_name = file_name.split("/")[-1]
    files_already_in_store = files_in_store()
    return file_name in files_already_in_store


# Use this to debug the code
# Modify the code and start under the debugger
if __name__ == "__main__":
    files = files_in_store()
    print(f"Files in store: {files}")
