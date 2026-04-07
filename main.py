import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

from langchain.prompts import PromptTemplate;

load_dotenv()

app = FastAPI(title="RAG Application")

# --------------- state ---------------
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

vector_store = None  # will hold the FAISS index after upload

FAISS_INDEX_PATH = Path("faiss_index")

# --------------- helpers ---------------

def load_document(file_path: str):
    """Load a PDF or text file and return LangChain Documents."""
    # TODO 
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.txt'):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError("Only .pdf and .txt files are supported.")
    return loader.load()


def build_vector_store(documents):
    """Split documents into chunks and build a FAISS vector store."""
    # TODO 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(str(FAISS_INDEX_PATH))
    return store

def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    if FAISS_INDEX_PATH.exists():
        return FAISS.load_local(str(FAISS_INDEX_PATH), embeddings=embeddings, allow_dangerous_deserialization=True)
    return None

vector_store = load_vector_store()

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
        """
    )

    retriever = store.as_retriever(search_kwargs={"k": k})


    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    return qa_chain

# --------------- routes ---------------

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <body>
            <h2>RAG Application</h2>
            <form action="/upload" enctype="multipart/form-data" method="post">
                <input name="file" type="file"/>
                <input type="submit"/>
            </form>
            <br>
            <form action="/query" method="post">
                <input name="question" type="text"/>
                <input type="submit"/>
            </form>
        </body>
    </html>
    """


class QueryRequest(BaseModel):
    question: str


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global vector_store

    file_path = UPLOAD_DIR / file.filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        documents = load_document(str(file_path))
        vector_store = build_vector_store(documents)
        return {"message": "Document uploaded and processed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
async def query_document(req: QueryRequest):
    global vector_store

    if vector_store is None:
        raise HTTPException(status_code=400, detail="No document uploaded yet.")

    try:
        qa_chain = get_qa_chain(vector_store)
        result = qa_chain.invoke({"query": req.question})

        return {
            "answer": result["result"],
            "sources": [doc.page_content[:200] for doc in result["source_documents"]]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))