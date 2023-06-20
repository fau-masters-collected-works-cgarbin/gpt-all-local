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
