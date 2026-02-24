import os
import sys
from pathlib import Path
from source_to_text.pdf_to_text import extract_text_from_pdfs
from source_to_text.video_to_text import extract_text_from_videos
from chunking.recursive_chunker import chunk_recursive
from embed.embed import embed_chunks_to_db
from retrieve.retrieve import retrieve
from evaluate.evaluate_response import evaluate_response
from linkedIn_post.linkedin_post import create_linkedin_post
from linkedIn_post.linkedin_self_analyze import generate_self_analyze_linkedin_post
from utils.log_utils import print_header, print_usage
from agent_state import AgentState
from workflow import implement_agentic_workflow
from config import (
    PDF_PATTERN,
    VIDEO_PATTERN,
    CHUNKED_DIR,
    VIDEO_TRANSCRIPT_DIR,
    PERSIST_DIR
)

# Imports for evaluation functionality
from typing import List

def run_agentic_query(query: str):
    """Execute the agentic workflow with a query"""
    print_header("RUNNING AGENTIC RAG WORKFLOW")

    # Get the compiled workflow
    workflow_app = implement_agentic_workflow()
    if not workflow_app:
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
        final_state = workflow_app.invoke(initial_state)

        # Extract information for evaluation
        question = final_state["question"]
        answer = final_state.get("generation", "No answer generated")

        # Build context string for evaluation
        context_parts = []
        if final_state.get("documents"):
            context_parts.append("Local docs: " + str(len(final_state["documents"])) + " documents")
        if final_state.get("web_search_results"):
            # Include a sample of web search results for context
            web_results = final_state.get("web_search_results", "")
            context_parts.append(f"Web search results: {web_results[:200]}...")
            #context_parts.append("Web search results available")
        context = ", ".join(context_parts) if context_parts else "No context"

        # Display results
        search_strategy = final_state.get("search_decision", "N/A")
        print(f"\n🔍 Search Strategy: {search_strategy}")
        print(f"🎯 Final Answer:\n{answer}")

        # Evaluate the response
        print("\n📊 EVALUATING RESPONSE...")
        evaluation = evaluate_response(question, answer, context)

        print(f"\n📊 EVALUATION RESULTS:")
        print(f"   Accuracy: {evaluation.accuracy}/5")
        print(f"   Relevance: {evaluation.relevance}/5") 
        print(f"   Clarity: {evaluation.clarity}/5")
        print(f"   Reasoning: {evaluation.reasoning}")

        # Calculate overall score
        overall_score = (evaluation.accuracy + evaluation.relevance + evaluation.clarity) / 3
        print(f"   Overall Score: {overall_score:.1f}/5")

        return final_state, evaluation

    except Exception as e:
        print(f"Workflow error: {e}")
        return None, None


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
    elif len(sys.argv) > 1 and sys.argv[1] in ["--linkedin", "-li"]:
        create_linkedin_post()
        sys.exit(0)
    elif len(sys.argv) > 1 and sys.argv[1] in ["--inkeldin_self_analyze", "-lsa"]:
        print_header("GENERATING SELF-ANALYZED LINKEDIN POST")
        try:
            generate_self_analyze_linkedin_post()
        except Exception as e:
            print(f"Error generating self-analyzed LinkedIn post: {e}")
        sys.exit(0)
    
    if is_first_run():
        run_full_pipeline(query)
    else:
        # Use agentic workflow instead of simple retrieval
        if query:
            run_agentic_query(query)
        else:
            print("No query provided. Please provide a query for agentic workflow.")
            print("💡 Tip: You can also generate a LinkedIn post with --linkedin or --linkedin_self_analyze")
