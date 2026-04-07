import streamlit as st
import requests

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="RAG Application", page_icon="📄", layout="centered")

st.title("📄 RAG — Ask Your Document")

# ── Upload Section ──
st.header("1. Upload a Document")
st.caption("Supports **.pdf** and **.txt** files.")

uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt"])

if st.button("Upload", disabled=uploaded_file is None):
    with st.spinner("Uploading and processing…"):
        try:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            res = requests.post(f"{API_BASE}/upload", files=files)
            if res.ok:
                st.success(res.json().get("message", "Upload successful!"))
            else:
                st.error(res.json().get("detail", "Upload failed."))
        except requests.ConnectionError:
            st.error("Cannot connect to the backend. Make sure the FastAPI server is running on port 8000.")

st.divider()

# ── Query Section ──
st.header("2. Ask a Question")

question = st.text_input("Your question", placeholder="e.g. What is the main topic of this document?")

if st.button("Ask", disabled=not question.strip()):
    with st.spinner("Thinking…"):
        try:
            res = requests.post(f"{API_BASE}/query", json={"question": question})
            if res.ok:
                data = res.json()
                st.subheader("Answer")
                st.write(data.get("answer", ""))

                sources = data.get("sources", [])
                if sources:
                    st.subheader("Sources")
                    for i, src in enumerate(sources, 1):
                        st.info(f"**[{i}]** {src}")
            else:
                st.error(res.json().get("detail", "Something went wrong."))
        except requests.ConnectionError:
            st.error("Cannot connect to the backend. Make sure the FastAPI server is running on port 8000.")
