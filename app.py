import os
import PyPDF2
import uuid
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
model = SentenceTransformer('all.MiniLM-L6-v2')

def read_pdf(file_name) :
    pdf_file = open(file_name, 'rb')
    reader = PyPDF2.PdfReader(pdf_file)
    pages = []
    for page_num, page_content in enumerate(reader.pages):
        text = page_content.extract_text()
        if text:
            pages.append({
                "text" : text,
                "page": page_num + 1,
                "source": file_name
            })
    return pages

def split_text(pages):
    chunks = []
    for page_data in pages:
        text = page_data["text"]
        
        for i in range(0, len(text), 500):
            chunk = text[i:i+500]
            chunks.append({
                "text" : chunk,
                "page" : page_data["page"],
                "source": page_data["source"]
            })
    return chunks

def embed_chunks(chunks):
    embed = []
    for chunk in chunks:
        embed.append(
            chunk["text"] 
        )
    
    vectors = model.encode(embed)
    return vectors

def store_vectors(chunks,vectors):
    client = chromadb.Client()
    ids = []
    documents = []
    metadatas = []
    
    for chunk in chunks:
        ids.append(str(uuid.uuid4()))
        documents.append(chunk["text"])
        metadatas.append({
            "page" : chunk["page"],
            "source" : chunk["source"]
        })
        

    try:
        client.delete_collection("pdf_chunks")
    except Exception:
        pass
        
    collection = client.create_collection("pdf_chunks") 
    collection.add(
        
        embeddings=vectors,
        documents=documents,
        metadatas=metadatas,
        ids=ids
        
    )
    return collection


def search(question, collection):
    question_vector = model.encode(question)
    results = collection.query(
        query_embeddings=[question_vector.tolist()],
        n_results=3
    )
    if not results['documents'] or not results['documents'][0]:
        return None
    return results

def get_answer(question, results):
    context = " ".join(results['documents'][0])
    llm = ChatGroq(model="llama-3.3-70b-versatile")    
    
    