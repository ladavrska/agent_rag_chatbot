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