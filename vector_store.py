import os
import re
from typing import List, Tuple
import json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

HR_MANUAL_PATH = "data.txt"
VECTOR_STORE_PATH = "vector_store"
METADATA_PATH = "metadata.json"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
MIN_CHUNK_LENGTH = 150

# Define embeddings globally so all functions can use it
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

def load_and_clean_text_file(file_path: str) -> str:
    """Load text from file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Remove extra whitespaces and normalize spacing
    text = re.sub(r'\s+', ' ', text)
    
    # Fix spacing around punctuation (ensure single space after punctuation)
    text = re.sub(r'([.!?])\s+', r'\1 ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\:\;\-\(\)]', '', text)
    
    # Clean up any remaining multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing punctuation and spaces from the entire text
    text = re.sub(r'^[\s\.\,\!\?\:\;\-]+|[\s\.\,\!\?\:\;\-]+$', '', text)
    
    return text.strip()


def create_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    )
    
    chunks = text_splitter.split_text(text)
    
    filtered_chunks = []
    for chunk in chunks:
        cleaned_chunk = chunk.strip()
        
        cleaned_chunk = re.sub(r'^[\s\.\,\!\?\:\;\-]+', '', cleaned_chunk)
        cleaned_chunk = cleaned_chunk.strip()
        
        if len(cleaned_chunk) >= MIN_CHUNK_LENGTH:
            filtered_chunks.append(cleaned_chunk)
    
    return filtered_chunks

def create_vector_store():
    cleaned_text = load_and_clean_text_file(HR_MANUAL_PATH)

    text_chunks = create_chunks(cleaned_text)
    print(f"Created {len(text_chunks)} text chunks")
    print("Initializing embedding model...")

    print("Creating documents...")
    documents = []
    for i, chunk in enumerate(text_chunks):
        doc = Document(
            page_content=chunk,
            metadata={"chunk_id": i, "source": "data"}
        )
        documents.append(doc)

    print("Building FAISS vector store...")
    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )

    print(f"Vector store created with {vector_store.index.ntotal} vectors")

    print("Saving vector store...")
    vector_store.save_local(VECTOR_STORE_PATH)
    print(f"Vector store saved to: {VECTOR_STORE_PATH}")

def query_vector_store(query: str, top_k: int = 5) -> List[Tuple[str, float]]:
    """
    Query the vector store and return top-k most similar chunks.
    
    Args:
        query: The search query
        top_k: Number of top results to return
        
    Returns:
        List of tuples (chunk_text, similarity_score)
    """
    try:
        loaded_vector_store = FAISS.load_local(
            VECTOR_STORE_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        results_with_scores = loaded_vector_store.similarity_search_with_score(query, k=top_k)
        results = [(doc.page_content, float(score)) for doc, score in results_with_scores]
        
        return results
    
    except Exception as e:
        print(f"Error querying vector store: {e}")
        return []

if __name__ == "__main__":
    if not os.path.exists(VECTOR_STORE_PATH):
        create_vector_store()

    print("\nTesting vector store...")

    test_query = "Who said quantum phenomena based hardware is needed?"
    print(f"Query: '{test_query}'")

    results = query_vector_store(test_query, top_k=3)

    if results:
        for i, (chunk, score) in enumerate(results, 1):
            print(f"\n{i}. Score: {score:.3f}")
            print(f"Text: {chunk[:200]}...")
    else:
        print("No results found")
