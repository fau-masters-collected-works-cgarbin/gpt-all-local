"""Vector store functions to abstract it from the rest of the code."""
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.vectorstores import VectorStore
import constants


def store() -> VectorStore:
    """Return an initialized vector store."""
    embeddings = HuggingFaceEmbeddings(model_name=constants.EMBEDDINGS_MODEL_NAME)
    return Chroma(persist_directory=constants.STORAGE_DIR, embedding_function=embeddings,
                  client_settings=constants.CHROMA_SETTINGS)


def files_in_store(store: VectorStore) -> list[str]:
    """Return the list of files in the store."""
    # We need to be careful here because we are making assumptions about the format of external data
    try:
        metadata = store.get(["metadatas"])["metadatas"]
        set_of_files = set()
        for file_name in metadata:
            set_of_files.add(file_name["source"])
        return list(set_of_files)
    except:  # noqa: E722
        return []
