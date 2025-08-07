from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

app = FastAPI()

# Define the path to the uploaded PDF
PDF_PATH = "Arogya Sanjeevani Policy - CIN - U10200WB1906GOI001713 1.pdf"
INDEX_PATH = "faiss_index"

# ✅ Model for asking questions
class QuestionRequest(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "HuggingFace PDF QnA System Running ✅"}

# ✅ Step 1: Embed the uploaded PDF
@app.post("/embed-local-pdf/")
def embed_uploaded_pdf():
    try:
        if not os.path.exists(PDF_PATH):
            raise HTTPException(status_code=404, detail="PDF file not found")

        loader = PyMuPDFLoader(PDF_PATH)
        pages = loader.load_and_split()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(pages)

        embeddings = HuggingFaceEmbeddings()
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(INDEX_PATH)

        return {"message": f"Document embedded successfully with {len(chunks)} chunks"}

    except Exception as e:
        return {"error": str(e)}

# ✅ Step 2: Query the document
@app.post("/query/")
def query_uploaded_pdf(request: QuestionRequest):
    try:
        if not os.path.exists("faiss_index"):
            raise HTTPException(status_code=404, detail="Index not found. Run /embed-local-pdf first.")

        embeddings = HuggingFaceEmbeddings()
        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

        results = db.similarity_search(request.question, k=3)

        return {
            "question": request.question,
            "answers": [r.page_content for r in results]
        }

    except Exception as e:
        return {"error": str(e)}

