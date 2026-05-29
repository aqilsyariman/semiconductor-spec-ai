import os
import PyPDF2
import uuid
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from flashrank import Ranker, RerankRequest

load_dotenv()
model = SentenceTransformer('all-MiniLM-L6-v2')
llm = ChatGroq(model="llama-3.3-70b-versatile")

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
        
        for i in range(0, len(text), 1000):
            chunk = text[i:i+1000]
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
        n_results=5
    )
    if not results['documents'] or not results['documents'][0]:
        return None
    return results

def get_answer(question, reranked):
    context = ""
    for result in reranked:
        source = result['meta']['source']
        page = result['meta']['page']
        context += f"\n[From {source}, Page {page}]:\n{result['text']}\n"
    
    prompt = f"""You are a semiconductor technical expert.
    Use this context to answer the question. 
    If a direct answer exists in the context, use it. Do not calculate or estimate.
    Context: {context}
    Question: {question}
    At the end cite all sources used."""
    answer = llm.invoke(prompt) 
    return answer.content

def rerank(question,results):    
    ranker = Ranker()
    passages = []

    for i, doc in enumerate(results['documents'][0]):    
        passages.append({
            "id" : i,
            "text": doc,
            "meta": {
                "source": results['metadatas'][0][i]['source'],
                "page": results['metadatas'][0][i]['page']}
        })

    request = RerankRequest(query=question, passages=passages)
    reranked = ranker.rerank(request)
    
    return reranked