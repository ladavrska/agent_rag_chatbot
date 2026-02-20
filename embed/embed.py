from langchain_chroma import Chroma
from config import (CHUNKS_PATTERN, PERSIST_DIR, COLLECTION_NAME, EMBEDDING_MODEL)
from utils.log_utils import log_embedding_summary
from utils.embed_utils import load_chunk_files, initialize_ollama_embeddings

def embed_chunks_to_db():

    # 1.Load documents from multiple chunk files
    documents = load_chunk_files(CHUNKS_PATTERN)

    # 2. Add nomic-embed-text prefixes for documents
    print("Adding search_document prefixes for nomic-embed-text model...")
    for doc in documents:
        doc.page_content = f"search_document: {doc.page_content}"

    # 3. Initialize Ollama Embeddings
    embeddings = initialize_ollama_embeddings(model_name=EMBEDDING_MODEL)

    # 4. Create and persist the local Chroma DB with batch processing
    print("Creating vector database...")
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION_NAME
    )

    # 5. Display summary by source file
    log_embedding_summary(documents)
    
