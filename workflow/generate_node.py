from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from agent_state import AgentState


def generate_node(state: AgentState, llm):
    """Action: Generates the final answer using available sources."""
    print("---GENERATING---")

    try:
        # Gather available sources
        local_docs = state.get("documents", [])
        web_results = state.get("web_search_results", "")
        question = state["question"]

        # Build context from available sources with accurate labels
        context_parts = []

        if local_docs:
            context_parts.append(f"Technical Documentation:\n{chr(10).join(local_docs)}")

        if web_results and web_results.strip():
            context_parts.append(f"Web Search Results:\n{web_results}")

        # Create unified prompt
        if context_parts:
            context = f"\n\n{chr(10).join(context_parts)}"

            gen_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful AI assistant. Answer the user's question using the provided sources. "
                         "Clearly indicate which source you're using in your response. "
                         "If multiple sources are available, synthesize information from all sources. "
                         "If the sources don't contain relevant information, say so clearly."),
                ("human", "Question: {question}\n\nAvailable Sources: {context}\n\nAnswer:")
            ])

            prompt_input = {"question": question, "context": context}
        else:
            # No sources available
            gen_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful AI assistant."),
                ("human", "I don't have access to relevant information to answer: {question}\n\n"
                         "Please provide a helpful response explaining this limitation.")
            ])

            prompt_input = {"question": question}

        # Generate response
        gen_chain = gen_prompt | llm
        response = gen_chain.invoke(prompt_input)

        print("Generation completed successfully")

        return {
            "question": state["question"],
            "documents": state["documents"],
            "generation": response.content,
            "loop_count": state["loop_count"],
            "relevance_grade": state.get("relevance_grade", ""),
            "web_search_results": state.get("web_search_results", ""),
            "search_decision": state.get("search_decision", "")
        }

    except Exception as e:
        print(f"Generation error: {e}")
        return {
            "question": state["question"],
            "documents": state["documents"],
            "generation": f"Error generating response: {str(e)}",
            "loop_count": state["loop_count"],
            "relevance_grade": state.get("relevance_grade", ""),
            "web_search_results": state.get("web_search_results", ""),
            "search_decision": state.get("search_decision", "")
        }