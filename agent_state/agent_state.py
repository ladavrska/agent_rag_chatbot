from typing import List, TypedDict

class AgentState(TypedDict):
    """State for the agentic RAG workflow."""
    question: str
    documents: List[str]
    generation: str
    loop_count: int
    relevance_grade: str
    web_search_results: str
    search_decision: str
