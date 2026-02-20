# Database Configuration
PERSIST_DIR = "./embed_db"
COLLECTION_NAME = "embed_chunks"

# Extraction to text
PDF_PATTERN = "./sources/*.pdf"
VIDEO_PATTERN = "./sources/*.mp4"
VIDEO_TRANSCRIPT_DIR = "./video_transcripts/"
CHUNKED_DIR = "./chunked"
CHUNKED_TRANSCRIPTS_FILE = "chunked_transcripts.json"

# Model Configuration
EMBEDDING_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3"
TEMPERATURE = 0

# Retrieval Configuration
RETRIEVER_TYPE = "mmr"  # or "similarity"
RETRIEVER_K = 5  # Number of documents to retrieve
RETRIEVER_FETCH_K = 10  # Fetch more candidates for diversity
RETRIEVER_LAMBDA_MULT = 0.7  # Balance between relevance and diversity

# Chunking Configuration (if you want to centralize all config)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CHUNKS_PATTERN = "./chunked/chunked_*.json"

SYSTEM_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "RAG in provided context refers to Retrieval-Augmented Generation. "
    "If you don't know the answer, just say that you don't know. "
    "\n\n"
    "{context}"
)
