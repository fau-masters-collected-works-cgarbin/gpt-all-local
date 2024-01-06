"""Streamlit App for the project."""

import tempfile

import streamlit as st

import ingest
import retrieve
import vector_store

st.set_page_config(page_title="RAG prototype", page_icon="🔎", layout="wide")
st.title("Retrieval Augmented Generation (RAG) prototype")
st.subheader("A prototype for RAG with all pieces running locally")
st.image("pics/solution-part2-similarity search-no letters.drawio.png")


def show_files_in_store():
    """Show which files are already in the store."""
    st.subheader("Files already in the store")
    files_in_store = vector_store.files_in_store()
    if len(files_in_store) == 0:
        st.write("No documents in the store")
    else:
        with st.expander("Click to see documents (files) already in the store"):
            files_in_store = sorted(files_in_store)
            for file_name in files_in_store:
                st.write(file_name)


def add_file_to_store():
    """Upload a file to the store."""
    st.subheader("Add documents (files) to the store")
    file_to_upload = st.file_uploader("Upload a file (if it's not in the store yet)")
    if file_to_upload is None:
        return

    if vector_store.file_stored(file_to_upload.name):
        st.write(f"'{file_to_upload.name}' is already in the store")
        return

    with st.spinner("Adding document to the store (please be patient - everything is running on your computer)..."):
        # Read the file and save to a local temporary directory (ingestion requires a local file)
        file_contents = file_to_upload.read()
        with tempfile.TemporaryDirectory() as temp_dir:
            with open(f"{temp_dir}/{file_to_upload.name}", "wb") as temp_file:
                temp_file.write(file_contents)
                ingest.ingest(temp_dir)


def answer_question():
    """Let the user ask a question and retrieve the answer."""
    st.subheader("Ask a question")
    question = st.text_input(
        "Question", placeholder="Ask a question  - press Enter to submit", label_visibility="hidden"
    )
    if question is None or question == "":
        return

    with st.spinner("Retrieving answer (please be patient - everything is running on your computer)..."):
        answer, docs = retrieve.query(question)
        st.write(f"Answer: {answer}")
        with st.expander("Click to show/hide the chunks used to answer the question"):
            cols = st.columns(len(docs))
            for i, doc in enumerate(docs):
                with cols[i]:
                    chunk = doc.page_content  # type: ignore
                    file = doc.metadata["source"].split("/")[-1]  # type: ignore
                    st.markdown(f"**Chunk {i+1} with {len(chunk)} characters, from {file}**")
                    st.write(chunk)


show_files_in_store()
add_file_to_store()
answer_question()
