from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from agent_state import AgentState


class QueryRewrite(BaseModel):
    """Improved query for better document retrieval."""
    rewritten_query: str = Field(description="Reformulated query optimized for document retrieval")
    changes_made: str = Field(description="Summary of what changes were made and why")
    confidence: float = Field(description="Confidence that the rewrite will improve results (0-1)")


def rewrite_query_node(state: AgentState):
    """Reflection: Self-corrects the search query for better results."""
    print("---REWRITING QUERY---")

    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a query optimization expert. Rewrite the user's question to improve document retrieval. Focus on key concepts, use technical terminology when appropriate, and make the query more specific."),
        ("human", "Original question: {question}\n\nRewrite this query for better document retrieval:")
    ])

    try:
        rewrite_llm = ChatOllama(model="llama3", format="json", temperature=0)
        structured_rewrite_llm = rewrite_llm.with_structured_output(QueryRewrite)
        rewrite_chain = rewrite_prompt | structured_rewrite_llm

        rewrite_result = rewrite_chain.invoke({"question": state["question"]})

        print(f"Original: {state['question']}")
        print(f"Rewritten: {rewrite_result.rewritten_query}")
        print(f"Changes: {rewrite_result.changes_made}")
        print(f"Confidence: {rewrite_result.confidence:.2f}")

        # Only use rewritten query if confidence is high
        final_question = rewrite_result.rewritten_query if rewrite_result.confidence > 0.7 else state["question"]

        return {
            "question": final_question,
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