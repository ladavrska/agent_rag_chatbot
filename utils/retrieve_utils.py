from utils.log_utils import print_header

def execute_query(rag_chain, query: str):
    """Execute a single query and display results"""
    print_header("RAG QUERY EXECUTION")
    print(f"\nQuery: {query}\n")

    # Add search_query prefix for nomic-embed-text model
    prefixed_query = f"search_query: {query}"
    
    try:
        response = rag_chain.invoke({"input": prefixed_query})
        print(f"Answer: {response['answer']}")
        
        # Optional: Show source documents
        if 'context' in response:
            print(f"\n📚 Sources used: {len(response['context'])} documents")
            
    except Exception as e:
        print(f"❌ Error executing query: {e}")
        

def run_interactive_mode(rag_chain):
    """Run interactive mode allowing multiple queries"""
    print_header("INTERACTIVE RAG MODE")
    print("Enter your queries (type 'quit' or 'exit' or Ctrl+Cto stop):")
    print("You can also try sample queries by typing 'samples'\n")
    
    while True:
        try:
            user_query = input("Query: ").strip()
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            elif user_query.lower() == 'samples':
                show_sample_queries()
                continue
            elif not user_query:
                print("Please enter a query or 'quit' to exit.")
                continue
                
            print(f"\n🔍 Searching for: {user_query}")
	    # Add search_query prefix for nomic-embed-text model
            prefixed_query = f"search_query: {user_query}"
            response = rag_chain.invoke({"input": prefixed_query})
            print(f"\n💡 Answer: {response['answer']}\n")
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            
def show_sample_queries():
    """Display sample queries for user reference"""
    sample_queries = [
        "What is the ColPali revolution?",
        "What is the difference between standard retrieval and the ColPali approach?",
        "What are RAG production best practices Do's?",
        "Why is hybrid search better than vector-only search?",
        "When to use hybrid search?",
        "What are hybrid search Do's for production?",
        "What are the production \"Do's\" for RAG?",
        "What is PaliGemma-3B Vision language model?"
    ]
    
    print("\n📝 Sample queries you can try:")
    for i, query in enumerate(sample_queries, 1):
        print(f"  {i}. {query}")
    print()


