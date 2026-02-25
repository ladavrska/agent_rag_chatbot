
from dataclasses import Field
from typing import List
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from linkedIn_post.linkedin_post import LinkedInPost
from utils.log_utils import print_header

class AgentSelfAnalysis(BaseModel):
    """Agent's self-analysis for social media."""
    key_capabilities: List[str] = Field(description="List of main agent capabilities")
    technologies_used: List[str] = Field(description="Key technologies and frameworks")
    learning_highlights: List[str] = Field(description="Key learning points from building this")
    challenges_overcome: List[str] = Field(description="Technical challenges solved")

def analyze_agent_capabilities() -> AgentSelfAnalysis:
    """Let the agent analyze its own capabilities for social media."""

    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are analyzing your own AI agent capabilities for a LinkedIn post. 
        Be specific about technical features and learning outcomes.
        Focus on what makes this RAG system unique (agentic workflows, reflection, structured outputs)."""),
        ("human", """Analyze this RAG chatbot agent's capabilities:

        Features:
        - Agentic workflow with LangGraph state management
        - Semantic search strategy routing (local vs web)
        - Document relevance grading with confidence scores
        - Query rewriting for better retrieval
        - Structured outputs with Pydantic validation
        - Multi-source information synthesis
        - Reflection and loop detection

        Provide a technical but accessible analysis for LinkedIn audience.""")
    ])

    llm = ChatOllama(model="llama3", format="json", temperature=0.3)
    structured_llm = llm.with_structured_output(AgentSelfAnalysis)
    analysis_chain = analysis_prompt | structured_llm

    return analysis_chain.invoke({})

def display_linkedin_post(result: LinkedInPost):
    """Display the LinkedIn post content."""
    if result:
        print("🔥 SELF-ANALYZED LINKEDIN POST READY:")
        print("=" * 60)
        print(result.post_content)
        print()
        if hasattr(result, 'hashtags') and result.hashtags:
            print(f"📱 Suggested hashtags: {result.hashtags}")
            print()
        if hasattr(result, 'call_to_action') and result.call_to_action and result.call_to_action.strip():
            print(f"💡 Call to action: {result.call_to_action}")
            print()
        print("=" * 60)
        print("✅ Ready to copy and paste to LinkedIn!")
    else:
        print("⚠️ No post content generated")

def generate_self_analyze_linkedin_post() -> LinkedInPost:
    """Generate and display LinkedIn post based on agent self-analysis."""
    
    print_header("GENERATING SELF-ANALYZED LINKEDIN POST")

    # First, analyze the agent
    analysis = analyze_agent_capabilities()
    
    linkedin_prompt = ChatPromptTemplate.from_messages([
        ("system", """Create a LinkedIn post based on the agent analysis. 
        Make it personal, professional, and engaging.
        Mention @Ciklum naturally and highlight the learning journey."""),
        ("human", """Based on this agent analysis, create a LinkedIn post:

        Capabilities: {capabilities}
        Technologies: {technologies} 
        Learning highlights: {learning}
        Challenges: {challenges}

        Requirements:
        - 5-7 sentences
        - Include @Ciklum mention
        - Professional but personal tone
        - Highlight both technical achievement and learning journey""")
    ])

    llm = ChatOllama(model="llama3", format="json", temperature=0.7)
    structured_llm = llm.with_structured_output(LinkedInPost)
    post_chain = linkedin_prompt | structured_llm

    result = post_chain.invoke({
        "capabilities": ", ".join(analysis.key_capabilities),
        "technologies": ", ".join(analysis.technologies_used),
        "learning": ", ".join(analysis.learning_highlights),
        "challenges": ", ".join(analysis.challenges_overcome)
    })

    # Display the result
    display_linkedin_post(result)

    return result