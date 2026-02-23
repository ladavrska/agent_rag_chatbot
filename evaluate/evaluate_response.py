from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Any

class EvaluationMetrics(BaseModel):
    """Evaluation scores for chatbot responses."""
    accuracy: int = Field(description="How factually correct is the answer? (1-5 scale)")
    relevance: int = Field(description="How well does the answer address the question? (1-5 scale)")
    clarity: int = Field(description="How clear and understandable is the answer? (1-5 scale)")
    reasoning: str = Field(description="Brief explanation of the scores")

def evaluate_response(question: str, answer: str, context: str = "") -> EvaluationMetrics:
    """Evaluate response quality using LLM-as-judge."""

    eval_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert evaluator for AI chatbot responses. 
        Rate the response on three dimensions using a 1-5 scale:

        ACCURACY (1-5): How factually correct is the information?
        - 5: Completely accurate, no errors
        - 3: Mostly accurate with minor issues
        - 1: Contains significant factual errors
        
        RELEVANCE (1-5): How well does the answer address the question?
        - 5: Directly answers the question completely
        - 3: Addresses the question but misses some aspects
        - 1: Doesn't answer the question asked
        
        CLARITY (1-5): How clear and understandable is the answer?
        - 5: Very clear, well-structured, easy to understand
        - 3: Generally clear with some confusion
        - 1: Confusing, hard to understand
        """),
        ("human", """Question: {question}
        
        Answer: {answer}
        
        Context Used: {context}
        
        Please evaluate this response:""")
    ])

    try:
        eval_llm = ChatOllama(model="llama3", format="json", temperature=0)
        structured_eval_llm = eval_llm.with_structured_output(EvaluationMetrics)
        eval_chain = eval_prompt | structured_eval_llm
        
        evaluation = eval_chain.invoke({
            "question": question,
            "answer": answer,
            "context": context[:500] if context else "No context available"
        })
        
        return evaluation
        
    except Exception as e:
        print(f"Evaluation error: {e}")
        return EvaluationMetrics(
            accuracy=0, relevance=0, clarity=0, 
            reasoning=f"Evaluation failed: {str(e)}"
        )