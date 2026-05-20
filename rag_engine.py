import os
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv(override=True)

def configure_gemini():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "your_key_here":
        raise ValueError("GEMINI_API_KEY is not configured. Add your real Gemini API key to the .env file.")
    genai.configure(api_key=api_key)

chroma_client = None
collection = None

def get_collection():
    global chroma_client, collection
    if collection is not None:
        return collection
        
    if not os.path.exists("./chroma_db"):
        raise RuntimeError("Run python ingest.py first to embed your documents.")
        
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    try:
        collection = chroma_client.get_collection(name="rag_docs")
        return collection
    except Exception:
        raise RuntimeError("Collection 'rag_docs' not found. Run python ingest.py first to embed your documents.")

def retrieve(query: str, top_k: int = 4) -> list[dict]:
    configure_gemini()
    col = get_collection()
    
    count = col.count()
    print(f"[RAG] Number of chunks in collection before querying: {count}")
    if count == 0:
        raise RuntimeError("No documents found. Run python ingest.py first.")
    
    # Embed the user query
    result = genai.embed_content(
        model="models/embedding-001",
        content=query,
        task_type="retrieval_query"
    )
    query_embedding = result['embedding']
    print(f"[RAG] Query embedding dimensions: {len(query_embedding)}")
    
    # Query ChromaDB for top_k similar chunks
    db_results = col.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    chunks = []
    if db_results['documents'] and len(db_results['documents']) > 0:
        docs = db_results['documents'][0]
        metadatas = db_results['metadatas'][0]
        distances = db_results['distances'][0]
        
        for i in range(len(docs)):
            chunk_text = docs[i]
            source = metadatas[i].get("source", "Unknown")
            score = distances[i]
            
            if i == 0:
                print(f"[RAG] Top result distance score: {score}")
                print(f"[RAG] Top result text snippet: {chunk_text[:100]}...")
                
            chunks.append({
                "text": chunk_text,
                "source": source,
                "score": score
            })
            
    return chunks

def format_context(chunks: list[dict]) -> str:
    context_parts = []
    for chunk in chunks:
        context_parts.append(f"[Source: {chunk['source']}]\n{chunk['text']}")
    return "\n\n".join(context_parts)

def collection_stats() -> dict:
    try:
        col = get_collection()
        count = col.count()
        all_data = col.get(include=['metadatas'])
        sources = set()
        if all_data and all_data['metadatas']:
            for meta in all_data['metadatas']:
                sources.add(meta.get("source", "Unknown"))
                
        return {
            "total_chunks": count,
            "sources": list(sources)
        }
    except Exception:
        return {
            "total_chunks": 0,
            "sources": []
        }
