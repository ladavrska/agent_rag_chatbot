from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from utils.log_utils import print_header

class LinkedInPost(BaseModel):
    """Structured LinkedIn post content."""
    post_content: str = Field(description="The main LinkedIn post text (5-7 sentences)")
    hashtags: str = Field(description="Relevant hashtags for the post")
    call_to_action: str = Field(description="Optional call-to-action or engagement prompt")


def generate_linkedin_post() -> LinkedInPost:
    """Generate a LinkedIn post about the RAG chatbot agent."""

    linkedin_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a professional LinkedIn content creator. Write a concise, engaging LinkedIn post about an AI RAG chatbot you built.

        Requirements:
        - 5-7 sentences maximum
        - Professional but approachable tone
        - Explain what the agent does and key technologies used
        - Mention it was built as part of Ciklum AI Academy
        - Include @Ciklum mention naturally
        - Focus on learning journey and technical achievement

        Key technologies to mention: RAG (Retrieval Augmented Generation), LangGraph, vector databases, document retrieval, agentic workflows
        """),
        ("human", """Create a LinkedIn post about my AI RAG chatbot agent that:
        - Performs intelligent document retrieval and web search
        - Uses agentic workflows with LangGraph for reasoning and reflection
        - Can rewrite queries and grade document relevance
        - Built with semantic search and structured outputs
        - Created during Ciklum AI Academy program
        
        Make it sound authentic and highlight the learning experience.""")
    ])

    try:
        llm = ChatOllama(model="llama3", format="json", temperature=0.7)  # Slightly higher temp for creativity
        structured_llm = llm.with_structured_output(LinkedInPost)
        post_chain = linkedin_prompt | structured_llm
        
        post = post_chain.invoke({})
        return post

    except Exception as e:
        print(f"LinkedIn post generation error: {e}")
        return LinkedInPost(
            post_content="Error generating LinkedIn post",
            hashtags="",
            call_to_action=""
        )
        
def create_linkedin_post():
    """Generate and display a LinkedIn post about the agent."""
    print_header("GENERATING LINKEDIN POST")
    
    post = generate_linkedin_post()
    
    print("🔥 LINKEDIN POST READY:")
    print("=" * 60)
    print(post.post_content)
    print()
    if post.hashtags:
        print(f"📱 Suggested hashtags: {post.hashtags}")
        print()
    if post.call_to_action:
        print(f"💡 Call to action: {post.call_to_action}")
        print()
    print("=" * 60)
    print("✅ Ready to copy and paste to LinkedIn!")


