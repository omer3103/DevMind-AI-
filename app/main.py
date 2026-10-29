
import os
import time
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi import UploadFile, File
from pypdf import PdfReader
from io import BytesIO
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
load_dotenv()

vector_store = None

app = FastAPI(title="DevMind AI")

class ChatRequest(BaseModel):
    question: str
 
class RagRequest(BaseModel):
    question: str

@app.get("/")
def home():
    return {
        "message": "Welcome to DevMind AI!",
        "status": "running"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/chat")
def chat(request: ChatRequest):
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GOOGLE_API_KEY is missing from .env"
        )

    try:
        model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    google_api_key=api_key,
    temperature=0.3,
    timeout=60,
    max_retries=0
)


        start_time = time.time()

        response = model.invoke(request.question)

        print(f"Gemini response time: {time.time() - start_time:.2f} seconds")

        return {"answer": response.content}

    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini request failed: {str(e)}"
        )



    
@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF file"
        )

    try:
        content = await file.read()
        pdf = PdfReader(BytesIO(content))

        text = "\n".join(
            page.extract_text() or ""
            for page in pdf.pages
        )
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_text(text)

        global vector_store

        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="No extractable text found in this PDF"
            )

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="GOOGLE_API_KEY is missing from .env"
            )

        embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=api_key
        )

        vector_store = FAISS.from_texts(
            texts=chunks,
            embedding=embeddings
        )

        return {
                     "filename": file.filename,
                    "pages": len(pdf.pages),
                    "characters_extracted": len(text),
                   "chunks_created": len(chunks),
                   "faiss_index_created": vector_store is not None

        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read PDF: {str(e)}"
        )

    
@app.post("/ask-rag")
def ask_rag(request: RagRequest):
    if vector_store is None:
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF first"
        )

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GOOGLE_API_KEY is missing from .env"
        )

    try:
        relevant_docs = vector_store.similarity_search(
            request.question,
            k=3
        )

        context = "\n\n".join(
            doc.page_content for doc in relevant_docs
        )

        model = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash",
            google_api_key=api_key,
            temperature=0.3,
            timeout=60,
            max_retries=0
        )

        prompt = f"""
You are a document question-answering assistant.

Answer the question using only the provided context.
If the answer is not present in the context, say:
"I couldn't find that information in the uploaded document."

Context:
{context}

Question:
{request.question}

Answer:
"""

        response = model.invoke(prompt)

        return {
            "question": request.question,
            "answer": response.content,
            "chunks_retrieved": len(relevant_docs)
        }

    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"RAG request failed: {str(e)}"
        )
