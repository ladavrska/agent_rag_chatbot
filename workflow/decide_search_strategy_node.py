from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from agent_state import AgentState


class SearchStrategy(BaseModel):
    """Plan for how to retrieve information based on the user question."""

    decision: Literal["web_search", "local_only"] = Field(
        description="The source to use. Use 'web_search' for current events, news, or general knowledge. Use 'local_only' for technical RAG and AI concepts."
    )
    reasoning: str = Field(description="Brief explanation of why this source was chosen.")


def decide_search_strategy(state: AgentState):
    """Decide whether to use local docs or web search using semantic LLM reasoning."""
    print("---DECIDING SEARCH STRATEGY---")

    # Create a structured decision prompt
    decision_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert search router.
        Your goal is to decide if a question should be answered using local technical documentation or the live web.

        LOCAL DOCUMENTS focus on: RAG architecture, vector databases (Pinecone, Milvus),
        LLM orchestration (LangChain, LangGraph), and AI engineering patterns.

        WEB SEARCH is for: Current events (2024-2026), stock prices, weather,
        general non-technical knowledge, or information about specific real-world people/entities."""),
        ("human", "{question}")
    ])

    try:
        # Use structured output with the LLM
        llm = ChatOllama(model="llama3", format="json", temperature=0)
        structured_llm = llm.with_structured_output(SearchStrategy)
        decision_chain = decision_prompt | structured_llm

        # Get structured decision from LLM
        strategy = decision_chain.invoke({"question": state["question"]})

        print(f"Search strategy decision: {strategy.decision}")
        print(f"Reasoning: {strategy.reasoning}")

        decision = strategy.decision

    except Exception as e:
        print(f"Routing Error, falling back to local: {e}")
        decision = "local_only"

    return {
        "question": state["question"],
        "documents": state.get("documents", []),
        "generation": state.get("generation", ""),
        "loop_count": state["loop_count"],
        "relevance_grade": state.get("relevance_grade", ""),
        "web_search_results": state.get("web_search_results", ""),
        "search_decision": decision
    }