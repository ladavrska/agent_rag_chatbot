import os
import sys
from pathlib import Path
from source_to_text.pdf_to_text import extract_text_from_pdfs
from source_to_text.video_to_text import extract_text_from_videos
from chunking.recursive_chunker import chunk_recursive
from embed.embed import embed_chunks_to_db
from retrieve.retrieve import retrieve
from utils.log_utils import print_header, print_usage
from config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    PDF_PATTERN,
    RETRIEVER_K,
    VIDEO_PATTERN,
    CHUNKED_DIR,
    VIDEO_TRANSCRIPT_DIR,
    PERSIST_DIR,
    SYSTEM_PROMPT
)

# Agentic Reasoning/Reflection Imports
from typing import List, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama

from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Agentic WebSearch Implementation
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool

# --- 1. DEFINE THE STATE ---
class AgentState(TypedDict):
    question: str
    documents: List[str]
    generation: str
    loop_count: int
    relevance_grade: str


def implement_agentic_workflow():
    # --- 2. DEFINE THE NODES (Reasoning & Action) ---
    print("Defining the nodes for the retrieval and generation process...")
    llm = ChatOllama(model="llama3", temperature=0)

    try:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        vector_db = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )
        retriever = vector_db.as_retriever(search_kwargs={"k": RETRIEVER_K})
    except Exception as e:
        print(f"Error loading vector DB: {e}")
        return None

    def retrieve_node(state: AgentState):
        """Retrieves documents from your Chroma vector_db."""
        print("---RETRIEVING---")
        query = f"search_query: {state['question']}"  # Add prefix for nomic-embed-text
        
        try:
            documents = retriever.invoke(query)
            doc_contents = [doc.page_content for doc in documents]
            return {
                "documents": doc_contents,
                "loop_count": state.get("loop_count", 0) + 1,
                "question": state["question"],
                "generation": state.get("generation", ""),
                "relevance_grade": state.get("relevance_grade", "")
            }
        except Exception as e:
            print(f"Retrieval error: {e}")
            return {
                "documents": [],
                "loop_count": state.get("loop_count", 0) + 1,
                "question": state["question"],
                "generation": state.get("generation", ""),
                "relevance_grade": state.get("relevance_grade", "")
            }

    def grade_documents_node(state: AgentState):
        """Reasoning: Checks if retrieved docs are actually relevant."""
        print("---CHECKING RELEVANCE---")

        # Create grading prompt
        grade_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a grader assessing relevance of retrieved documents to a user question. "
                      "If the document contains keywords or information related to the user question, grade it as relevant. "
                      "Give a binary score 'yes' or 'no' to indicate whether the document is relevant."),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}")
        ])

        grade_chain = grade_prompt | llm

        # Grade each document
        relevant_docs = []
        for doc in state["documents"]:
            try:
                grade = grade_chain.invoke({"document": doc, "question": state["question"]})
                if "yes" in grade.content.lower():
                    relevant_docs.append(doc)
            except Exception as e:
                print(f"Grading error: {e}")
                continue

        print(f"Found {len(relevant_docs)} relevant documents out of {len(state['documents'])}")

        # Return updated state with filtered documents and relevance grade
        return {
            "documents": relevant_docs,
            "question": state["question"],
            "generation": state.get("generation", ""),
            "loop_count": state["loop_count"],
            "relevance_grade": "relevant" if relevant_docs else "irrelevant"
        }

    def generate_node(state: AgentState):
        """Action: Generates the final answer using local and/or web sources."""
        print("---GENERATING---")

        try:
            # Combine local documents and web results
            local_context = "\n\n".join(state.get("documents", []))
            web_context = state.get("web_search_results", "")

            # Create combined context
            combined_context = ""
            if local_context:
                combined_context += f"Local Knowledge Base:\n{local_context}\n\n"
            if web_context:
                combined_context += f"Web Search Results:\n{web_context}"

            # Use combined context in system prompt
            formatted_system_prompt = SYSTEM_PROMPT.format(context=combined_context)

            gen_prompt = ChatPromptTemplate.from_messages([
                ("system", formatted_system_prompt),
                ("human", "{question}")
            ])

            gen_chain = gen_prompt | llm

            response = gen_chain.invoke({"question": state["question"]})
            return {
                "generation": response.content,
                "documents": state["documents"],
                "question": state["question"],
                "loop_count": state["loop_count"],
                "relevance_grade": state.get("relevance_grade", ""),
                "web_search_results": state.get("web_search_results", ""),
                "search_decision": state.get("search_decision", "")
            }
        except Exception as e:
            print(f"Generation error: {e}")
            return {
                "generation": "Sorry, I couldn't generate an answer.",
                "documents": state["documents"],
                "question": state["question"],
                "loop_count": state["loop_count"],
                "relevance_grade": state.get("relevance_grade", ""),
                "web_search_results": state.get("web_search_results", ""),
                "search_decision": state.get("search_decision", "")
            }

    def rewrite_query_node(state: AgentState):
        """Reflection: Self-corrects the search query for better results."""
        print("---REWRITING QUERY---")

        # Create query rewriting prompt
        rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a query re-writer. Your task is to re-write the user question to improve retrieval. "
                      "Look at the input and try to reason about the underlying semantic intent/meaning."),
            ("human", "Here is the initial question: \n\n {question} \n\n "
                     "Formulate an improved question that would retrieve better documents:")
        ])

        rewrite_chain = rewrite_prompt | llm

        try:
            response = rewrite_chain.invoke({"question": state["question"]})
            rewritten_question = response.content
            print(f"Original: {state['question']}")
            print(f"Rewritten: {rewritten_question}")

            return {
                "question": rewritten_question,
                "documents": state["documents"],
                "generation": state.get("generation", ""),
                "loop_count": state["loop_count"],
                "relevance_grade": state.get("relevance_grade", "")
            }
        except Exception as e:
            print(f"Rewrite error: {e}")
            return state  # Return original state if error

    def web_search_node(state: AgentState):
        """Web search when local documents are insufficient."""
        print("---WEB SEARCHING---")

        try:
            # Create a search-optimized query
            search_query = f"search_query: {state['question']}"
            web_results = web_search.invoke({"query": search_query})

            print(f"Web search completed for: {state['question']}")

            return {
                "question": state["question"],
                "documents": state["documents"],
                "generation": state.get("generation", ""),
                "loop_count": state["loop_count"],
                "relevance_grade": state.get("relevance_grade", ""),
                "web_search_results": web_results,
                "search_decision": state.get("search_decision", "")
            }
        except Exception as e:
            print(f"Web search error: {e}")
            return {
                "question": state["question"],
                "documents": state["documents"],
                "generation": state.get("generation", ""),
                "loop_count": state["loop_count"],
                "relevance_grade": state.get("relevance_grade", ""),
                "web_search_results": f"Web search failed: {str(e)}",
                "search_decision": state.get("search_decision", "")
            }

    # Add decision node for search strategy
    def decide_search_strategy(state: AgentState):
        """Decide whether to use local docs, web search, or both."""
        print("---DECIDING SEARCH STRATEGY---")

        # Create decision prompt
        decision_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a search strategist. Analyze the user question and determine the best search approach. "
                      "Consider: Is this about recent events, current data, or general knowledge that might need web search? "
                      "Or is it likely covered in local documents? "
                      "Respond with: 'local_only', 'web_search', or 'both'"),
            ("human", "User question: {question}\n\n"
                     "Available local documents found: {doc_count} documents\n"
                     "Relevance of local docs: {relevance}")
        ])

        decision_chain = decision_prompt | llm

        try:
            response = decision_chain.invoke({
                "question": state["question"],
                "doc_count": len(state.get("documents", [])),
                "relevance": state.get("relevance_grade", "unknown")
            })

            decision = response.content.lower().strip()

            # Ensure valid decision
            if decision not in ["local_only", "web_search", "both"]:
                decision = "local_only"  # Default fallback

            print(f"Search strategy decision: {decision}")

            return {
                "question": state["question"],
                "documents": state["documents"],
                "generation": state.get("generation", ""),
                "loop_count": state["loop_count"],
                "relevance_grade": state.get("relevance_grade", ""),
                "web_search_results": state.get("web_search_results", ""),
                "search_decision": decision
            }
        except Exception as e:
            print(f"Decision error: {e}")
            return {
                "question": state["question"],
                "documents": state["documents"],
                "generation": state.get("generation", ""),
                "loop_count": state["loop_count"],
                "relevance_grade": state.get("relevance_grade", ""),
                "web_search_results": state.get("web_search_results", ""),
                "search_decision": "local_only"  # Fallback
            }

    # --- 3. BUILD THE AGENTIC GRAPH ---
    workflow = StateGraph(AgentState)

    # Add all nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("decide_strategy", decide_search_strategy)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("rewrite", rewrite_query_node)

    # Define the workflow edges
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_edge("grade_documents", "decide_strategy")

    # Add conditional logic for search strategy
    def route_search_strategy(state):
        """Route based on search strategy decision"""
        decision = state.get("search_decision", "local_only")

        if decision == "web_search":
            return "web_search"
        elif decision == "both":
            return "web_search"  # Will combine with local docs in generation
        else:  # local_only
            return "check_generation"

    def check_generation_readiness(state):
        """Enhanced decision logic for generation vs rewrite"""
        # Prevent infinite loops
        if state.get("loop_count", 0) >= 3:
            print("Max iterations reached, proceeding to generation...")
            return "generate"

        relevance_grade = state.get("relevance_grade", "irrelevant")
        search_decision = state.get("search_decision", "local_only")

        # If we have relevant docs OR web search results, generate
        if (relevance_grade == "relevant" or
            (search_decision in ["web_search", "both"] and state.get("web_search_results"))):
            return "generate"

        return "rewrite"

    workflow.add_conditional_edges("decide_strategy", route_search_strategy)
    workflow.add_edge("web_search", "generate")

    # Add a routing node for generation decision
    workflow.add_node("check_generation", lambda state: state)
    workflow.add_conditional_edges("check_generation", check_generation_readiness)

    workflow.add_edge("rewrite", "retrieve")  # The Reflection Loop
    workflow.add_edge("generate", END)

    # Compile and return the workflow
    app = workflow.compile()
    return app

def run_agentic_query(query: str):
    """Execute the agentic workflow with a query"""
    print_header("RUNNING ENHANCED AGENTIC RAG WORKFLOW")

    # Get the compiled workflow
    app = implement_agentic_workflow()
    if not app:
        print("Failed to initialize agentic workflow")
        return

    # Define initial state
    initial_state = {
        "question": query,
        "documents": [],
        "generation": "",
        "loop_count": 0,
        "relevance_grade": "",
        "web_search_results": "",
        "search_decision": ""
    }

    try:
        # Execute the workflow
        final_state = app.invoke(initial_state)

        print(f"\n🎯 Final Answer: {final_state['generation']}")
        print(f"🔄 Iterations: {final_state['loop_count']}")
        print(f"🔍 Search Strategy: {final_state.get('search_decision', 'N/A')}")

    except Exception as e:
        print(f"❌ Workflow execution error: {e}")


def run_full_pipeline(query: str = None):
    """Run the complete pipeline for first-time setup"""
    print_header("FIRST RUN DETECTED - Running Full Pipeline")
    
    # 1: Extract PDFs to Text
    pdf_to_text()
    
    # 2: Extract text from videos  
    video_to_text()
    
    # 3: Chunk video transcripts
    chunk_video_transcripts()
    
    # 4: Embed chunks to DB
    embed()
    
    # 5: Test retrieval
    retrieve_query(query)
    
    
def run_retrieval_only(query: str = None):
    """Run only retrieval for subsequent runs"""
    retrieve_query(query)

def pdf_to_text():
    print("\tLoading and extracting text from PDF...")
    
    print_header("Starting PDF to Text conversion...")
    successful, failed = extract_text_from_pdfs(PDF_PATTERN, CHUNKED_DIR)
    
    print(f"\nPDF to Text conversion completed!")
    print(f"Successfully converted: {len(successful)} files")
    print(f"Failed conversions: {len(failed)} files")
    
    return successful, failed

def video_to_text():
    """Extract text from multiple videos"""
    print_header("Starting transcribing Video to Text...")
    extract_text_from_videos(VIDEO_PATTERN, VIDEO_TRANSCRIPT_DIR)

def chunk_video_transcripts():
    """Chunk video transcripts using recursive chunker"""
    print_header("Starting chunking of video transcripts...")
    chunk_recursive(input_dir=VIDEO_TRANSCRIPT_DIR, output_dir=CHUNKED_DIR)

def embed():
    """Embed chunks to vector DB"""
    print_header("Starting embedding chunks to vector DB...")
    embed_chunks_to_db()

def retrieve_query(query: str = None):
    """Retrieve from embedded DB"""
    print_header("Starting retrieval from embedded DB...")
    
    if query:
        retrieve(query)
    else:
        # Use default query or prompt user
        print("No query provided. Using interactive mode...")
        retrieve()

def is_first_run():
    """Check if this is the first run by looking for embed DB and processed files"""
    embed_db_exists = os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR)
    chunked_files_exist = os.path.exists(CHUNKED_DIR) and any(Path(CHUNKED_DIR).glob("chunked_*.json"))
    
    return not (embed_db_exists and chunked_files_exist)

def get_query_from_args():
    """Get query from command line arguments"""
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:])
    return None


if __name__ == "__main__":
    
    query = get_query_from_args()
    
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help"]:
        print_usage()
        sys.exit(0)
    
    if is_first_run():
        run_full_pipeline(query)
    else:
        # Use agentic workflow instead of simple retrieval
        if query:
            run_agentic_query(query)
        else:
            print("No query provided. Please provide a query for agentic workflow.")
