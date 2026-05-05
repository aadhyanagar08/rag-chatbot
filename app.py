import streamlit as st
import tempfile
import os
from ingest import ingest_pdf
from query import answer_question

st.set_page_config(page_title="Document Q&A", layout="centered")
st.title("Document Q&A Chatbot")
st.caption("Upload a PDF and ask questions about it")

if "ingested" not in st.session_state:
    st.session_state.ingested = False

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file and not st.session_state.ingested:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("Reading and indexing your document..."):
        ingest_pdf(tmp_path)
        st.session_state.ingested = True
        os.unlink(tmp_path)

    st.success("Document indexed. Ask your questions below.")

if st.session_state.ingested:
    question = st.text_input("Ask a question about your document:")

    if question:
        with st.spinner("Searching document..."):
            result = answer_question(question)

        st.markdown("### Answer")
        st.write(result["answer"])

        with st.expander("Show source chunks used"):
            for i, chunk in enumerate(result["source_chunks"]):
                st.markdown(f"**Chunk {i+1}:**")
                st.write(chunk)
                st.divider()