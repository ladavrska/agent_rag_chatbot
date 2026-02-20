from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from config import (PERSIST_DIR, COLLECTION_NAME, EMBEDDING_MODEL, RETRIEVER_TYPE, RETRIEVER_K, RETRIEVER_FETCH_K, RETRIEVER_LAMBDA_MULT, SYSTEM_PROMPT)
from utils.retrieve_utils import execute_query, run_interactive_mode

def retrieve(query: str = None):
    """Retrieve answers from embedded PDF chunks using RAG with Ollama LLM"""
    
    # 1. Load the vector database
    print("Loading vector database...")
    try:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        vector_db = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )
        print(f"✓ Vector DB loaded from {PERSIST_DIR}")
    except Exception as e:
        print(f"❌ Error loading vector DB: {e}")
        raise

    # 2. Initialize Llama 3 via Ollama with context length limit
    
    """ A LangChain wrapper for Ollama's chat models that enables conversational AI capabilities"""
    llm = ChatOllama(
        model="llama3", # Llama 3 has a context window of ~8K tokens
        temperature=0
    )
    
     # 3. Retrieval chain setup
    system_prompt = (SYSTEM_PROMPT + "\n\n {context}")

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    retriever = vector_db.as_retriever(
        search_type=RETRIEVER_TYPE,
        search_kwargs={
            "k": RETRIEVER_K,  # Number of documents to retrieve
            "fetch_k": RETRIEVER_FETCH_K,  # Fetch more candidates to ensure diversity
            "lambda_mult": RETRIEVER_LAMBDA_MULT  # Balance between relevance and diversity
        }
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    # 4. Handle query input
    if query:
        # Use provided query
        execute_query(rag_chain, query)
    else:
        # Interactive mode - allow multiple queries
        run_interactive_mode(rag_chain)
    
    
if __name__ == "__main__":
    retrieve()
