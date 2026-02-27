"""Utility functions for workflow routing and decision logic."""


def route_search_strategy(state):
    """Route based on search strategy decision"""
    decision = state.get("search_decision", "local_only")
    print(f"Routing decision: {decision}")

    if decision == "web_search":
        return "web_search"
    else:  # local_only or both
        return "retrieve"


def check_local_results(state):
    """Check if local results are sufficient or need fallback"""
    relevance_grade = state.get("relevance_grade", "irrelevant")
    relevance_confidence = state.get("relevance_confidence", 0.0)
    loop_count = state.get("loop_count", 0)

    print(f"Checking local results - relevance: {relevance_grade}, confidence: {relevance_confidence:.2f}, loops: {loop_count}")

    # Prevent infinite loops
    if loop_count >= 3:
        print("Max iterations reached, proceeding to generation...")
        return "generate"

    if relevance_grade == "relevant":
        print("Decision: Documents are relevant → GENERATE")
        return "generate"
    
    # If documents are irrelevant with HIGH confidence, skip rewrite and go to web search
    elif relevance_grade == "irrelevant" and relevance_confidence > 0.8:
        print(f"Decision: High confidence irrelevance ({relevance_confidence:.2f}) → WEB SEARCH")
        return "web_search"
    
    # If first attempt and not high confidence irrelevance, try rewriting
    elif loop_count < 1:
        print("Decision: First attempt with low confidence irrelevance → REWRITE")
        return "rewrite"
    
    # After rewrite attempt, fallback to web search
    else:
        print("Decision: Already tried rewrite → WEB SEARCH")
        return "web_search"