import streamlit as st
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

# --------------- config ---------------
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
FAISS_INDEX_PATH = Path("faiss_index")

# --------------- helpers ---------------

def load_document(file_path: str):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError("Only .pdf and .txt files are supported.")
    return loader.load()


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_vector_store(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    store = FAISS.from_documents(chunks, get_embeddings())
    store.save_local(str(FAISS_INDEX_PATH))
    return store


def load_vector_store():
    if FAISS_INDEX_PATH.exists():
        return FAISS.load_local(
            str(FAISS_INDEX_PATH),
            embeddings=get_embeddings(),
            allow_dangerous_deserialization=True,
        )
    return None


def get_qa_chain(store, k=3):
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
            You are a precise AI assistant.

            Use ONLY the provided context to answer.
            If unsure, say "I don't know".

            Context:
            {context}

            Question:
            {question}

            Give a concise answer:
        """,
    )

    retriever = store.as_retriever(search_kwargs={"k": k})
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )


# --------------- session state ---------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = load_vector_store()

# --------------- UI ---------------
st.set_page_config(page_title="RAG Application", page_icon="📄", layout="centered")
st.title("📄 RAG — Ask Your Document")

# ── Upload Section ──
st.header("1. Upload a Document")
st.caption("Supports **.pdf** and **.txt** files.")

uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt"])

if st.button("Upload", disabled=uploaded_file is None):
    with st.spinner("Uploading and processing…"):
        try:
            file_path = UPLOAD_DIR / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            documents = load_document(str(file_path))
            st.session_state.vector_store = build_vector_store(documents)
            st.success("Document uploaded and processed successfully!")
        except Exception as e:
            st.error(f"Error: {e}")

st.divider()

# ── Query Section ──
st.header("2. Ask a Question")

question = st.text_input("Your question", placeholder="e.g. What is the main topic of this document?")

if st.button("Ask", disabled=not question.strip()):
    if st.session_state.vector_store is None:
        st.error("No document uploaded yet. Please upload a document first.")
    else:
        with st.spinner("Thinking…"):
            try:
                qa_chain = get_qa_chain(st.session_state.vector_store)
                result = qa_chain.invoke({"query": question})

                st.subheader("Answer")
                st.write(result["result"])

                sources = result.get("source_documents", [])
                if sources:
                    st.subheader("Sources")
                    for i, doc in enumerate(sources, 1):
                        st.info(f"**[{i}]** {doc.page_content[:200]}")
            except Exception as e:
                st.error(f"Error: {e}")
