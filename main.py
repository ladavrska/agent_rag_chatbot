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
    web_search_results: str
    search_decision: str


def implement_agentic_workflow():
    # --- 2. DEFINE THE NODES (Reasoning & Action) ---
    print("Defining the nodes for the retrieval and generation process...")
    llm = ChatOllama(model="llama3", temperature=0)

    # Initialize web search tool
    web_search = DuckDuckGoSearchRun()

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
            doc_texts = [doc.page_content for doc in documents]
            print(f"Retrieved {len(doc_texts)} documents")

            return {
                "question": state["question"],
                "documents": doc_texts,
                "generation": state.get("generation", ""),
                "loop_count": state["loop_count"],
                "relevance_grade": state.get("relevance_grade", ""),
                "web_search_results": state.get("web_search_results", ""),
                "search_decision": state.get("search_decision", "")
            }
        except Exception as e:
            print(f"Retrieval error: {e}")
            return {
                "question": state["question"],
                "documents": [],
                "generation": state.get("generation", ""),
                "loop_count": state["loop_count"],
                "relevance_grade": state.get("relevance_grade", ""),
                "web_search_results": state.get("web_search_results", ""),
                "search_decision": state.get("search_decision", "")
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
            "relevance_grade": "relevant" if relevant_docs else "irrelevant",
            "web_search_results": state.get("web_search_results", ""),  # Add this line
            "search_decision": state.get("search_decision", "")  # Add this line
        }

    def generate_node(state: AgentState):
        """Action: Generates the final answer using local and/or web sources."""
        print("---GENERATING---")

        try:
            # Prepare context
            local_docs = state.get("documents", [])
            web_results = state.get("web_search_results", "")
            search_decision = state.get("search_decision", "")

            # Check what sources we actually have
            has_local_docs = local_docs and len(local_docs) > 0
            has_web_results = web_results and web_results.strip() != ""

            print(f"Generation context - Local docs: {len(local_docs)}, Web results: {bool(has_web_results)}, Decision: {search_decision}")

            # Create different prompts based on available sources
            if search_decision == "web_search" and has_web_results:
                # Pure web search - ignore local docs entirely
                gen_prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a helpful AI assistant. Answer the user's question using ONLY the web search results provided. "
                            "Do not mention 'provided context' or 'local documents'. "
                            "Base your answer entirely on the web search information."),
                    ("human", "Question: {question}\n\n"
                            "Web Search Results: {web_results}\n\n"
                            "Answer based on web search results:")
                ])

                response = gen_prompt.invoke({
                    "question": state["question"],
                    "web_results": web_results
                })

            elif has_local_docs and not has_web_results:
                # Pure local search
                gen_prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a helpful AI assistant. Answer the user's question using the provided documents. "
                            "If the documents don't contain relevant information, say so clearly."),
                    ("human", "Question: {question}\n\n"
                            "Documents: {documents}\n\n"
                            "Answer:")
                ])

                response = gen_prompt.invoke({
                    "question": state["question"],
                    "documents": "\n\n".join(local_docs)
                })

            elif has_local_docs and has_web_results:
                # Both sources available
                gen_prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a helpful AI assistant. Answer the user's question using both local documents and web search results. "
                            "Prioritize the most relevant and recent information. "
                            "Be clear about your sources."),
                    ("human", "Question: {question}\n\n"
                            "Local Documents: {documents}\n\n"
                            "Web Search Results: {web_results}\n\n"
                            "Answer using both sources:")
                ])

                response = gen_prompt.invoke({
                    "question": state["question"],
                    "documents": "\n\n".join(local_docs),
                    "web_results": web_results
                })

            else:
                # No useful sources
                response_content = f"I don't have enough information to answer the question '{state['question']}'. No relevant documents or web search results were found."
                response = type('Response', (), {'content': response_content})()

            # Use ChatOllama for the actual generation
            llm = ChatOllama(model="llama3", temperature=0)
            if hasattr(response, 'content'):
                final_response = response
            else:
                final_response = llm.invoke(response)

            print("Generation completed successfully")

            return {
                "question": state["question"],
                "documents": state["documents"],
                "generation": final_response.content,
                "loop_count": state["loop_count"],
                "relevance_grade": state.get("relevance_grade", ""),
                "web_search_results": state.get("web_search_results", ""),
                "search_decision": state.get("search_decision", "")
            }
        except Exception as e:
            print(f"Generation error: {e}")
            return {
                "question": state["question"],
                "documents": state["documents"],
                "generation": f"Error generating response: {str(e)}",
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
            rewritten_question = response.content.strip()
            print(f"Rewritten query: {rewritten_question}")

            return {
                "question": rewritten_question,
                "documents": [],
                "generation": state.get("generation", ""),
                "loop_count": state["loop_count"] + 1,
                "relevance_grade": "",
                "web_search_results": state.get("web_search_results", ""),
                "search_decision": state.get("search_decision", "")
            }
        except Exception as e:
            print(f"Query rewrite error: {e}")
            return {
                "question": state["question"],
                "documents": state["documents"],
                "generation": state.get("generation", ""),
                "loop_count": state["loop_count"] + 1,
                "relevance_grade": state.get("relevance_grade", ""),
                "web_search_results": state.get("web_search_results", ""),
                "search_decision": state.get("search_decision", "")
            }

    def web_search_node(state: AgentState):
        """Web search when local documents are insufficient."""
        print("---WEB SEARCHING---")

        try:
            # Perform web search
            search_results = web_search.run(state["question"])
            print(f"Web search results: {search_results[:300]}...")  # Show first 300 chars

            return {
                "question": state["question"],
                "documents": state["documents"],
                "generation": state.get("generation", ""),
                "loop_count": state["loop_count"],
                "relevance_grade": state.get("relevance_grade", ""),
                "web_search_results": search_results,
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
        """Decide whether to use local docs, web search, or both - BEFORE retrieval."""
        print("---DECIDING SEARCH STRATEGY---")

        question_lower = state["question"].lower()

        # Strong time-sensitive indicators (force web search immediately)
        strong_time_indicators = ["today", "now", "current", "latest", "recent", "this week", "this month",
                                "2024", "2025", "2026", "2027", "breaking", "just released", "new developments"]

        # Check for obvious web search keywords
        web_search_keywords = ["weather", "price", "stock", "news", "president", "prime minister",
                            "current events", "happening", "bitcoin", "cryptocurrency", "exchange rate"]

        # Technical/local keywords that suggest local documents might be useful
        local_keywords = ["rag", "retrieval augmented generation", "vector database", "embedding",
                        "chunking", "similarity search", "llm", "database design", "sql", "nosql",
                        "algorithm", "programming", "python", "machine learning", "ai model"]

        # Quick decision for obvious cases
        has_strong_time_indicators = any(indicator in question_lower for indicator in strong_time_indicators)
        has_web_keywords = any(keyword in question_lower for keyword in web_search_keywords)
        has_local_keywords = any(keyword in question_lower for keyword in local_keywords)

        if has_strong_time_indicators or has_web_keywords:
            print(f"Immediate web search decision due to keywords: {[k for k in strong_time_indicators + web_search_keywords if k in question_lower]}")
            decision = "web_search"
        elif has_local_keywords:
            print(f"Local search decision due to technical keywords: {[k for k in local_keywords if k in question_lower]}")
            decision = "local_only"
        else:
            # Use LLM for ambiguous cases
            print("Using LLM for search strategy decision...")

            decision_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a search strategist. Analyze the user question and determine the best search approach.\n\n"
                        "Use 'web_search' for:\n"
                        "- Real-time information (weather, prices, news, current events)\n"
                        "- Current people/positions (presidents, CEOs, current affairs)\n"
                        "- Time-sensitive data (latest, recent, today, now, current)\n"
                        "- Live data (stock prices, cryptocurrency, sports scores)\n"
                        "- Educational content (how to, what is, explain)\n"
                        "- Historical information\n"
                        "- General knowledge that doesn't change frequently\n"
                        "- Breaking news or recent developments\n\n"
                        "Use 'local_only' for:\n"
                        "- Technical concepts (RAG, databases, AI/ML, programming)\n"
                        "The local documents contain technical information about RAG, databases, and AI.\n\n"
                        "Respond with EXACTLY ONE word: 'web_search' or 'local_only'"),
                ("human", "Question: {question}\n\nDecision:")
            ])

            try:
                llm = ChatOllama(model="llama3", temperature=0)
                decision_chain = decision_prompt | llm

                response = decision_chain.invoke({"question": state["question"]})
                decision_text = response.content.lower().strip()

                if "web_search" in decision_text:
                    decision = "web_search"
                elif "local_only" in decision_text:
                    decision = "local_only"
                else:
                    # If unclear, default based on question type
                    if any(word in question_lower for word in ["who is", "what is the weather", "current", "now"]):
                        decision = "web_search"
                    else:
                        decision = "local_only"

            except Exception as e:
                print(f"LLM decision error: {e}")
                # Fallback logic
                if any(word in question_lower for word in ["who is", "weather", "current", "now", "today"]):
                    decision = "web_search"
                else:
                    decision = "local_only"

        print(f"Search strategy decision: {decision}")

        return {
            "question": state["question"],
            "documents": state.get("documents", []),
            "generation": state.get("generation", ""),
            "loop_count": state["loop_count"],
            "relevance_grade": state.get("relevance_grade", ""),
            "web_search_results": state.get("web_search_results", ""),
            "search_decision": decision
        }

    # --- 3. BUILD THE AGENTIC GRAPH ---
    workflow = StateGraph(AgentState)

    # Add all nodes
    workflow.add_node("decide_strategy", decide_search_strategy)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("rewrite", rewrite_query_node)

    # Start with strategy decision
    workflow.add_edge(START, "decide_strategy")

    # Route based on strategy
    def route_search_strategy(state):
        """Route based on search strategy decision"""
        decision = state.get("search_decision", "local_only")
        print(f"Routing decision: {decision}")

        if decision == "web_search":
            return "web_search"
        else:  # local_only or both
            return "retrieve"

    workflow.add_conditional_edges("decide_strategy", route_search_strategy)

    # Local document path
    workflow.add_edge("retrieve", "grade_documents")

    def check_local_results(state):
        """Check if local results are sufficient or need fallback"""
        relevance_grade = state.get("relevance_grade", "irrelevant")
        loop_count = state.get("loop_count", 0)

        print(f"Checking local results - relevance: {relevance_grade}, loops: {loop_count}")

        # Prevent infinite loops
        if loop_count >= 3:
            print("Max iterations reached, proceeding to generation...")
            return "generate"

        if relevance_grade == "relevant":
            return "generate"
        elif loop_count < 2:
            # Try rewriting query first
            return "rewrite"
        else:
            # If rewrite didn't help, try web search as fallback
            return "web_search"

    workflow.add_conditional_edges("grade_documents", check_local_results)

    # Both paths lead to generation
    workflow.add_edge("web_search", "generate")
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
        return

    # Define initial state
    initial_state = {
        "question": query,
        "documents": [],
        "generation": "",
        "loop_count": 0,
        "relevance_grade": "",
        "web_search_results": "",
        "search_decision": ""  # Initialize this
    }

    try:
        # Run the agentic workflow
        result = app.invoke(initial_state)

        # Debug: Print the final state keys to see what's available
        print(f"Final state keys: {list(result.keys())}")
        print(f"Search decision in result: {result.get('search_decision', 'NOT_FOUND')}")

        # Your logging code here - make sure it uses the correct key
        search_strategy = result.get("search_decision", "N/A")
        print(f"🔍 Search Strategy: {search_strategy}")
        print(f"🎯 Final Answer: {result.get('generation', 'No answer generated')}")

    except Exception as e:
        print(f"Workflow execution error: {e}")


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
