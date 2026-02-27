"""Agentic query execution functionality."""

from utils.log_utils import print_header
from workflow import create_agentic_workflow
from evaluate.evaluate_response import evaluate_response


def run_agentic_query(query: str):
    """Execute the agentic workflow with a query"""
    print_header("RUNNING AGENTIC RAG WORKFLOW")

    # Get the compiled workflow
    workflow_app = create_agentic_workflow()
    if not workflow_app:
        return

    # Define initial state
    initial_state = {
        "question": query,
        "documents": [],
        "generation": "",
        "loop_count": 0,
        "relevance_grade": "",
        "relevance_confidence": 0.0,
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