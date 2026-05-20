import os
import glob
from pypdf import PdfReader
import tiktoken
import google.generativeai as genai
import chromadb
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

def configure_gemini():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "your_key_here":
        raise ValueError("GEMINI_API_KEY is not configured. Add your real Gemini API key to the .env file.")
    genai.configure(api_key=api_key)

# Initialize Tokenizer (cl100k_base used by tiktoken)
tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    """Returns the token count of a given text."""
    return len(tokenizer.encode(text))

def recursive_character_split(text, chunk_size=500, overlap=50):
    """
    A simple recursive character splitter that respects token counts.
    It tries to split by paragraphs (\n\n), then lines (\n), then words ( ), and finally characters.
    """
    separators = ["\n\n", "\n", " ", ""]
    
    def _split_text(text, separators):
        # If the text fits in chunk_size, return it
        if count_tokens(text) <= chunk_size:
            return [text]
            
        separator = separators[-1]
        for sep in separators:
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                break
                
        # Split the text
        if separator != "":
            splits = text.split(separator)
        else:
            splits = list(text)
            
        good_splits = [s for s in splits if s]
        
        current_doc = []
        current_length = 0
        docs = []
        
        for s in good_splits:
            s_len = count_tokens(s)
            
            # If a single split is larger than chunk_size, split it further
            if s_len > chunk_size:
                if current_doc:
                    doc_text = separator.join(current_doc)
                    if doc_text.strip():
                        docs.append(doc_text)
                    current_doc = []
                    current_length = 0
                
                if len(separators) > 1:
                    sub_chunks = _split_text(s, separators[separators.index(separator) + 1:])
                    docs.extend(sub_chunks)
                else:
                    # Fallback for when we run out of separators (split exactly by tokens)
                    tokens = tokenizer.encode(s)
                    for i in range(0, len(tokens), chunk_size):
                        sub_s = tokenizer.decode(tokens[i:i + chunk_size])
                        if sub_s.strip():
                            docs.append(sub_s)
                continue
            
            sep_len = count_tokens(separator) if current_doc else 0
            
            # If adding this split exceeds chunk_size, save current doc
            if current_length + s_len + sep_len > chunk_size and current_doc:
                doc_text = separator.join(current_doc)
                if doc_text.strip():
                    docs.append(doc_text)
                
                # Handle overlap
                while current_doc:
                    current_doc_text = separator.join(current_doc)
                    if count_tokens(current_doc_text) > overlap:
                        removed = current_doc.pop(0)
                        current_length -= count_tokens(removed) + (count_tokens(separator) if current_doc else 0)
                    else:
                        break
                        
            current_doc.append(s)
            current_length += s_len + (count_tokens(separator) if len(current_doc) > 1 else 0)
            
        if current_doc:
            doc_text = separator.join(current_doc)
            if doc_text.strip():
                docs.append(doc_text)
                
        return docs

    return _split_text(text, separators)


def load_documents(directory_path="docs/"):
    """Loads PDF and TXT files from the specified directory."""
    docs = []
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created {directory_path}. Please add documents.")
        return docs

    for filepath in glob.glob(os.path.join(directory_path, "*")):
        filename = os.path.basename(filepath)
        if filename.startswith(".") or filename == "placeholder.txt":
            continue
            
        try:
            if filepath.lower().endswith(".txt"):
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                    if content.strip():
                        docs.append({"filename": filename, "content": content})
            elif filepath.lower().endswith(".pdf"):
                text = ""
                with open(filepath, "rb") as f:
                    reader = PdfReader(f)
                    for page in reader.pages:
                        extracted = page.extract_text()
                        if extracted:
                            text += extracted + "\n"
                if text.strip():
                    docs.append({"filename": filename, "content": text})
        except Exception as e:
            print(f"Warning: Failed to read {filename}. Error: {e}")
            
    return docs

def process_and_store():
    """Main pipeline to load, chunk, embed, and store documents."""
    configure_gemini()

    # Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    
    collection_name = "rag_docs"
    
    # Idempotency: clear existing collection if it exists
    try:
        chroma_client.delete_collection(name=collection_name)
    except Exception:
        pass # Collection might not exist yet
        
    collection = chroma_client.create_collection(name=collection_name)
    
    docs = load_documents()
    if not docs:
        print("No documents found in docs/ folder.")
        return
        
    total_chunks_stored = 0
    
    for doc in docs:
        filename = doc["filename"]
        content = doc["content"]
        
        # Split text into chunks
        chunks = recursive_character_split(content, chunk_size=500, overlap=50)
        if not chunks:
            continue
            
        embeddings = []
        ids = []
        metadatas = []
        documents = []
        
        for i, chunk in enumerate(chunks):
            try:
                # Embed each chunk
                result = genai.embed_content(
                    model="models/embedding-001",
                    content=chunk,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
                ids.append(f"{filename}_chunk_{i}")
                metadatas.append({"source": filename, "chunk_index": i})
                documents.append(chunk)
            except Exception as e:
                print(f"Warning: Failed to embed chunk {i} of {filename}. Error: {e}")
                
        if embeddings:
            # Store in ChromaDB
            collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Ingested: {filename} - {len(embeddings)} chunks")
            total_chunks_stored += len(embeddings)
            
    print(f"Total chunks stored: {total_chunks_stored}")

if __name__ == "__main__":
    process_and_store()
