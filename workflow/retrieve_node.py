from agent_state import AgentState


def retrieve_node(state: AgentState, retriever):
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