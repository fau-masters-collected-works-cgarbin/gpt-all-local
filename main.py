"""The entry point for the project.

Run with --help to see the available options.
"""

import argparse

import constants
import ingest
import logger
import retrieve

# Parse the command line arguments
ACTION_HELP = """The action to perform:
    - ingest: Ingest documents into the vector database.
    - retrieve: Ask questions on the documents with the help of the LLM.
"""
parser = argparse.ArgumentParser(
    description="Run an LLM on your own data.", formatter_class=argparse.RawTextHelpFormatter
)
parser.add_argument("action", choices=("ingest", "retrieve"), help=ACTION_HELP)
parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity.")
args = parser.parse_args()

log = logger.get_logger()
logger.set_verbose(args.verbose)

# Execute the action
if args.action == "ingest":
    log.info(f"Ingesting documents from '{constants.DATA_DIR}' into the vector database")
    ingest.ingest()
elif args.action == "retrieve":
    retrieve.check_requisites()
    while (question := input("\nEnter a question or 'exit': ")) != "exit":
        answer, documents = retrieve.query(question)
        if logger.VERBOSE:
            log.info("Chunks used to answer the question:")
            for i, document in enumerate(documents):
                chunk = document.page_content  # type: ignore
                file = document.metadata.get("source", "unknown").split("/")[-1]
                log.info(
                    f"   Chunk {i + 1} of {len(documents)} with {len(chunk)} characters,"
                    f" from file {file}\n{chunk[:50]} [...] {chunk[-50:]}"
                )

        print(f"\n\nAnswer: {answer}")
