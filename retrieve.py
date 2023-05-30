"""Retrieve information from the local data using an LLM.

At this point we a vector store populated with the contents of the local data. Now we want to retrieve information from
that local data using natural language. This involves the following steps:

1. Find the most similar document chunks to the query.
2. Build a prompt for the language model that contains the user query and the most similar document chunks.
3. Use the language model to generate the answer.

NOTE: This code assumes that the model has already been downloaded. See the README for instructions.

This code is heavily based on the ingest.py code from https://github.com/imartinez/privateGPT.
"""
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import GPT4All
from langchain.vectorstores import Chroma
import constants

# Set these variables only when explicilty asked to do so to not waste time and resources
EMBEDDINGS = None
VECTOR_STORE = None
MODEL = None


def prepare() -> None:
    """Prepare the environment for the retrieval."""
    global EMBEDDINGS  # pylint: disable=global-statement
    EMBEDDINGS = HuggingFaceEmbeddings(model_name=constants.EMBEDDINGS_MODEL_NAME)

    global VECTOR_STORE  # pylint: disable=global-statement
    VECTOR_STORE = Chroma(persist_directory=constants.STORAGE_DIR, embedding_function=EMBEDDINGS,
                          client_settings=constants.CHROMA_SETTINGS)

    global MODEL  # pylint: disable=global-statement
    MODEL = GPT4All(model=constants.MODEL_PATH, n_ctx=constants.MODEL_CONTEXT_WINDOW, backend='gptj', verbose=False)
