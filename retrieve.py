"""Retrieve information from the local data using an LLM.

At this point we a vector store populated with the contents of the local data. Now we want to retrieve information from
that local data using natural language. This involves the following steps:

1. Find the most similar document chunks to the query.
2. Build a prompt for the language model that contains the user query and the most similar document chunks.
3. Use the language model to generate the answer.

NOTE: This code assumes that the model has already been downloaded. See the README for instructions.

This code is heavily based on the ingest.py code from https://github.com/imartinez/privateGPT.
"""
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import GPT4All
from langchain.vectorstores import Chroma
import constants
import logger

# Set these variables only when explicilty asked to do so to not waste time and resources
_EMBEDDINGS = None
_MODEL = None
_RETRIEVER = None


def _prepared() -> bool:
    """Check if the environment is prepared for the retrieval."""
    return _EMBEDDINGS is not None and _RETRIEVER is not None and _MODEL is not None


def _prepare() -> None:
    """Prepare the environment for the retrieval."""
    if _prepared():
        return

    log = logger.get_logger()
    log.info("Preparing the environment for the retrieval")

    global _EMBEDDINGS  # pylint: disable=global-statement
    _EMBEDDINGS = HuggingFaceEmbeddings(model_name=constants.EMBEDDINGS_MODEL_NAME)
    log.debug("   Loaded embeddings model '%s'", constants.EMBEDDINGS_MODEL_NAME)

    global _MODEL  # pylint: disable=global-statement
    _MODEL = GPT4All(model=constants.MODEL_PATH, n_ctx=constants.MODEL_CONTEXT_WINDOW, backend='gptj',
                     verbose=logger.VERBOSE)
    log.debug("   Loaded language model from '%s' with context window %d", constants.MODEL_PATH,
              constants.MODEL_CONTEXT_WINDOW)

    # Build a retriever from the vector store, embeddings, and model
    vector_store = Chroma(persist_directory=constants.STORAGE_DIR, embedding_function=_EMBEDDINGS,
                          client_settings=constants.CHROMA_SETTINGS)
    vs_retriever = vector_store.as_retriever(search_kwargs={"k": constants.TARGET_SOURCE_CHUNKS})
    log.debug("   Loaded vector store from '%s'", constants.STORAGE_DIR)
    global _RETRIEVER  # pylint: disable=global-statement
    _RETRIEVER = RetrievalQA.from_chain_type(llm=_MODEL, chain_type="stuff", retriever=vs_retriever,
                                             return_source_documents=logger.VERBOSE)


def query(user_input: str) -> tuple[str, list[str]]:
    """Query the local data using the given query."""
    _prepare()

    log = logger.get_logger()
    log.info("Querying the local data with '%s'", query)
    query_result = _RETRIEVER(user_input)
    answer, source_documents = query_result["result"], query_result["source_documents"] if logger.VERBOSE else []
    return answer, source_documents
