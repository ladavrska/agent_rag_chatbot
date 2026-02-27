from langchain_community.tools import DuckDuckGoSearchRun

from agent_state import AgentState


def web_search_node(state: AgentState, web_search):
    """Web search when local documents are insufficient."""
    print("---WEB SEARCHING---")

    try:
        # Perform web search
        search_results = web_search.run(state["question"])
        print(f"Web search results: {search_results[:300]}...")

        return {
            "question": state["question"],
            "documents": state["documents"],
            "generation": state.get("generation", ""),
            "loop_count": state["loop_count"],
            "relevance_grade": state.get("relevance_grade", ""),
            "relevance_confidence": state.get("relevance_confidence", 0.0),
            "web_search_results": search_results,
            "search_decision": "web_search"
        }
    except Exception as e:
        print(f"Web search error: {e}")
        return {
            "question": state["question"],
            "documents": state["documents"],
            "generation": state.get("generation", ""),
            "loop_count": state["loop_count"],
            "relevance_grade": state.get("relevance_grade", ""),
            "relevance_confidence": state.get("relevance_confidence", 0.0),
            "web_search_results": f"Web search failed: {str(e)}",
            "search_decision": "web_search"
        }