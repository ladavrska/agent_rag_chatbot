from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from agent_state import AgentState


class DocumentRelevance(BaseModel):
    """Assessment of document relevance to user question."""
    is_relevant: bool = Field(description="True if document contains information relevant to the question")
    confidence: float = Field(description="Confidence score 0-1 for the relevance assessment")
    reasoning: str = Field(description="Brief explanation of why document is relevant/irrelevant")


def grade_documents_node(state: AgentState):
    """Reasoning: Checks if retrieved docs are actually relevant."""
    print("---CHECKING RELEVANCE---")

    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a strict document relevance grader. A document is only relevant if it DIRECTLY addresses the user's specific question.

        Mark as IRRELEVANT if:
        - The question is too vague or generic
        - The document only tangentially relates to the topic
        - The question lacks specific technical details that would help retrieval

        Mark as RELEVANT only if:
        - The document directly answers the specific question asked
        - The document contains concrete information that addresses the user's need"""),
        ("human", "Document: {document}\n\nUser Question: {question}\n\nAssess relevance:")
    ])

    try:
        # Use structured output for consistent grading
        grade_llm = ChatOllama(model="llama3", format="json", temperature=0)
        structured_grade_llm = grade_llm.with_structured_output(DocumentRelevance)
        grade_chain = grade_prompt | structured_grade_llm

        # Grade each document with structured output
        relevant_docs = []
        all_assessments = []
        
        for doc in state["documents"]:
            try:
                assessment = grade_chain.invoke({"document": doc, "question": state["question"]})
                all_assessments.append(assessment)

                print(f"Document relevance: {assessment.is_relevant} (confidence: {assessment.confidence:.2f})")
                print(f"Reasoning: {assessment.reasoning}")
                
                # Only include documents with high confidence relevance
                if assessment.is_relevant and assessment.confidence > 0.6:
                    relevant_docs.append(doc)

            except Exception as e:
                print(f"Grading error for document: {e}")
                continue

        print(f"Found {len(relevant_docs)} relevant documents out of {len(state['documents'])}")
        
        # Calculate average confidence for routing decisions
        if all_assessments:
            avg_confidence = sum(a.confidence for a in all_assessments) / len(all_assessments)
            print(f"Average assessment confidence: {avg_confidence:.2f}")
        else:
            avg_confidence = 0.0

        return {
            "documents": relevant_docs,
            "question": state["question"],
            "generation": state.get("generation", ""),
            "loop_count": state["loop_count"],
            "relevance_grade": "relevant" if relevant_docs else "irrelevant",
            "relevance_confidence": avg_confidence,
            "web_search_results": state.get("web_search_results", ""),
            "search_decision": state.get("search_decision", "")
        }

    except Exception as e:
        print(f"Document grading error: {e}")
        # Fallback to all documents if grading fails
        return {
            "documents": state["documents"],
            "question": state["question"],
            "generation": state.get("generation", ""),
            "loop_count": state["loop_count"],
            "relevance_grade": "unknown",
            "relevance_confidence": 0.0,
            "web_search_results": state.get("web_search_results", ""),
            "search_decision": state.get("search_decision", "")
        }