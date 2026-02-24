from typing import List, Literal
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Agentic WebSearch Implementation
from langchain_community.tools import DuckDuckGoSearchRun

from agent_state import AgentState
from workflow.retrieve_node import retrieve_node
from workflow.grade_documents_node import grade_documents_node
from workflow.generate_node import generate_node
from workflow.rewrite_query_node import rewrite_query_node
from workflow.web_search_node import web_search_node
from workflow.decide_search_strategy_node import decide_search_strategy
from workflow.workflow_utils import route_search_strategy, check_local_results
from config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    RETRIEVER_K,
    PERSIST_DIR
)


def implement_agentic_workflow():
    """Create and compile the agentic workflow graph."""

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

    # --- BUILD THE AGENTIC GRAPH ---
    workflow = StateGraph(AgentState)

    # Add all nodes
    workflow.add_node("decide_strategy", decide_search_strategy)
    workflow.add_node("retrieve", lambda state: retrieve_node(state, retriever))
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("web_search", lambda state: web_search_node(state, web_search))
    workflow.add_node("generate", lambda state: generate_node(state, llm))
    workflow.add_node("rewrite", lambda state: rewrite_query_node(state, llm)) # remove lambda, it needs llm = ChatOllama(model="llama3", format="json", temperature=0) Required for .with_structured_output() !!!

    # Start with strategy decision
    workflow.add_edge(START, "decide_strategy")

    workflow.add_conditional_edges("decide_strategy", route_search_strategy)

    # Local document path
    workflow.add_edge("retrieve", "grade_documents")

    workflow.add_conditional_edges("grade_documents", check_local_results)

    # Both paths lead to generation
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)

    # Compile and return the workflow
    workflow_app = workflow.compile()
    return workflow_app