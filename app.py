"""Streamlit App for the project."""

import tempfile
import streamlit as st
import ingest
import retrieve
import vector_store

st.header("Retrieval Augmented Generation (RAG)")
st.subheader("A prototype for RAG with all pieces running locally")
st.image("pics/solution-part2-similarity search-no letters.drawio.png")

st.subheader("Add documents (files) to the store")
files_in_store_section = st.empty()


def update_files_in_store_section():
    """Show which files are already in the store."""
    files_in_store = vector_store.files_in_store()
    if len(files_in_store) == 0:
        files_in_store_section.write("No documents in the store")
    else:
        with files_in_store_section.expander("Click to see documents (files) already in the store"):
            files_in_store = sorted(files_in_store)
            for file_name in files_in_store:
                st.write(file_name)
update_files_in_store_section()


def upload_file():
    """Upload a file to the store."""
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
                update_files_in_store_section()


file_to_upload = st.file_uploader("Upload a file (if it's not in the store yet)")
upload_file()


def answer_question():
    if question is None or question == "":
        return

    with st.spinner("Retrieving answer (please be patient - everything is running on your computer)..."):
        answer, docs = retrieve.query(question)
        st.write(f"Answer: {answer}")
        with st.expander("See the chunks used to answer the question"):
            for i, doc in enumerate(docs):
                chunk = doc.page_content
                file = doc.metadata["source"].split("/")[-1]
                st.markdown(f"**Chunk {i+1} with {len(chunk)} characters, from {file}**")
                st.write(chunk)
                st.divider()


st.subheader("Ask a question")
question = st.text_input("", placeholder="Ask a question  - press Enter to submit", label_visibility="collapsed")
answer_question()
