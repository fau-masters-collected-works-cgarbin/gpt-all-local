"""Retrieve information from the local data using an LLM.

At this point we a vector store populated with the contents of the local data. Now we want to retrieve information from
that local data using natural language. This involves the following steps:

1. Find the most similar document chunks to the query.
2. Build a prompt for the language model that contains the user query and the most similar document chunks.
3. Use the language model to generate the answer.

NOTE: This code assumes that the model has already been downloaded. See the README for instructions.

This code is heavily based on the ingest.py code from https://github.com/imartinez/privateGPT.
"""
import sys
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.llms import GPT4All
import constants
import logger
import vector_store

# Set these variables only when explicitly asked to do so to not waste time and resources
_MODEL = None
_RETRIEVER = None


def _prepared() -> bool:
    """Check if the environment is prepared for the retrieval."""
    return _RETRIEVER is not None and _MODEL is not None


def _prepare() -> None:
    """Prepare the environment for the retrieval."""
    if _prepared():
        return

    log = logger.get_logger()
    log.info("Preparing the environment for the retrieval")

    global _MODEL  # pylint: disable=global-statement
    _MODEL = GPT4All(model=str(constants.MODEL), n_batch=constants.MODEL_N_BATCH,
                     backend='gptj', verbose=logger.VERBOSE)
    log.debug("   Loaded language model from '%s'", constants.MODEL)

    # Build a retriever from the vector store, embeddings, and model
    db = vector_store.store()
    vs_retriever = db.as_retriever(search_kwargs={"k": constants.TARGET_SOURCE_CHUNKS})
    log.debug("   Loaded vector store from '%s'", constants.STORAGE_DIR)
    global _RETRIEVER  # pylint: disable=global-statement
    # TODO: test other options for `chain_type`
    _RETRIEVER = RetrievalQA.from_chain_type(llm=_MODEL, chain_type="stuff", retriever=vs_retriever,
                                             return_source_documents=True)


def check_requisites() -> None:
    """Check if the model has been downloaded."""
    if not constants.MODEL.is_file():
        log = logger.get_logger()
        log.error("Cannot find the model at '%s'. Please download it first (see the README).", constants.MODEL)
        sys.exit(1)


def query(user_input: str) -> tuple[Document, list[str]]:
    """Query the local data using the given query."""
    _prepare()

    log = logger.get_logger()
    log.info("Querying the local data with '%s'", user_input)
    query_result = _RETRIEVER(user_input)
    answer, source_documents = query_result["result"], query_result["source_documents"]
    return answer, source_documents


# Use this to debug the code
# Modify the question and start under the debugger
if __name__ == "__main__":
    logger.set_verbose(True)
    QUESTION = "What is a prompt"
    answer, documents = query(QUESTION)
    print(f"Question: {QUESTION}")
    print(f"Answer: {answer}")
    print(f"Documents: {documents}")
