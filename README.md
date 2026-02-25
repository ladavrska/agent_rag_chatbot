# Agentic RAG Chatbot

An advanced agentic RAG (Retrieval-Augmented Generation) system using LangGraph workflows with semantic routing, document grading, query rewriting, and multi-source information synthesis. Built for technical documentation and real-time information retrieval.

## Features

### 🤖 Agentic Workflow System
- **LangGraph State Management**: Advanced workflow orchestration with conditional routing
- **Semantic Search Strategy**: LLM-powered decision making for local vs web search
- **Document Relevance Grading**: Confidence-based document filtering with structured outputs
- **Query Rewriting**: Intelligent query refinement for better retrieval results
- **Reflection & Loop Detection**: Prevents infinite loops with smart fallback mechanisms

### 📊 Advanced Capabilities
- **Multi-Source Synthesis**: Combines local documents with real-time web search
- **LLM-as-Judge Evaluation**: Automated response quality assessment (accuracy, relevance, clarity)
- **Structured Outputs**: Pydantic validation for consistent LLM responses
- **LinkedIn Post Generation**: AI-powered content creation with self-analysis features
- **Web Search Integration**: DuckDuckGo search for current events and general knowledge

### 🔧 Data Processing Pipeline
- **PDF Processing**: Extract text from PDF files with intelligent chunking
- **Video Processing**: Extract and process video transcripts
- **Vector Embeddings**: ChromaDB storage with Ollama embeddings
- **Smart Pipeline**: Automatically detects first run vs. subsequent runs

## Usage

### Command Line Usage

```bash
# Agentic RAG queries (default workflow)
python3 main.py "What is LangGraph?"
python3 main.py How does semantic routing work in RAG systems

# LinkedIn content generation
python3 main.py --linkedin           # Generate LinkedIn post
python3 main.py -li                  # Short form

# AI self-analysis LinkedIn post
python3 main.py --linkedin_self_analyze    # Agent analyzes its own capabilities
python3 main.py -lsa                       # Short form

# Help and documentation
python3 main.py -h
python3 main.py --help
```

### Agentic Workflow Behavior

The system uses an intelligent agentic workflow that makes decisions about information retrieval:

#### 🧠 Semantic Search Strategy
- **Local Documents**: Technical topics (RAG, LangGraph, vector databases, AI engineering)
- **Web Search**: Current events, news, stock prices, general knowledge
- **Hybrid Approach**: Automatic fallback when local results are insufficient

#### 🔄 Workflow Steps
1. **Strategy Decision**: LLM determines best information source
2. **Document Retrieval**: Fetch relevant chunks from vector database  
3. **Relevance Grading**: Assess document quality and relevance
4. **Query Rewriting**: Refine queries for better results (if needed)
5. **Web Search Fallback**: Search the web if local docs are insufficient
6. **Response Generation**: Synthesize final answer with source attribution
7. **Quality Evaluation**: LLM-as-judge scoring (accuracy, relevance, clarity)

#### 📈 Evaluation Metrics
- **Accuracy**: Factual correctness (1-5 scale)
- **Relevance**: Response relevance to query (1-5 scale)  
- **Clarity**: Response clarity and readability (1-5 scale)
- **Overall Score**: Aggregated performance metric

### Pipeline Behavior

#### First Run Setup
When no vector database exists, the system executes the complete data processing pipeline:

1. **PDF to Text**: Extract text from all PDF files
2. **Video to Text**: Extract transcripts from video files  
3. **Recursive Chunking**: Split documents into optimal chunks
4. **Vector Embedding**: Create and store embeddings in ChromaDB
5. **Agentic Query**: Execute first query with full workflow

#### Subsequent Usage  
After initial setup, the system runs the optimized agentic workflow:

1. **Semantic Routing**: LLM decides information source strategy
2. **Conditional Retrieval**: Local documents or web search based on query type
3. **Quality Assessment**: Confidence scoring and relevance grading
4. **Intelligent Loops**: Query rewriting and fallback mechanisms
5. **Response Synthesis**: Multi-source information combination
6. **Automated Evaluation**: LLM-as-judge quality assessment

### Content Generation Features

#### LinkedIn Post Generation
- **Standard Posts**: Generate professional LinkedIn content
- **Self-Analysis Posts**: AI agent analyzes its own capabilities and learning journey
- **Technical Focus**: Emphasizes learning outcomes and technical achievements
- **@Ciklum Integration**: Naturally incorporates company mentions

## Requirements

### Core Dependencies
```bash
pip install -r requirements.txt
```

### Key Technologies
- **LangChain & LangGraph**: Agentic workflow orchestration
- **Ollama**: Local LLM integration (llama3 model)
- **ChromaDB**: Vector database with persistence
- **Pydantic**: Structured output validation
- **DuckDuckGo**: Web search integration

## Configuration

The system uses configuration settings from `config.py` for:
- PDF and video file patterns
- Model and embedding configurations  
- Database persistence directory
- Retrieval parameters (chunk size, overlap, k-value)

## Architecture

### 🏗️ Modular Workflow Design
```
workflow/
├── agentic_workflow.py           # Main workflow orchestration  
├── decide_search_strategy_node.py # Semantic routing logic
├── retrieve_node.py              # Local document retrieval
├── grade_documents_node.py       # Document relevance scoring
├── generate_node.py              # Response generation
├── rewrite_query_node.py         # Query refinement
├── web_search_node.py            # Web search integration
└── workflow_utils.py             # Routing utility functions
```

### 📊 State Management & Evaluation  
```
agent_state/
└── agent_state.py               # Centralized state definitions

evaluate/
└── evaluate_response.py         # LLM-as-judge evaluation
```

### 🎯 Content Generation
```
linkedIn_post/
├── linkedin_post.py             # Standard post generation
└── linkedin_self_analyze.py     # Self-analysis posts
```

## Project Structure

```
├── main.py                      # CLI orchestration & pipeline management
├── agentic_query.py             # Agentic workflow execution
├── config.py                    # Configuration settings
├── requirements.txt             # Python dependencies
├── agent_state/                 # State management
├── workflow/                    # Modular workflow nodes
├── evaluate/                    # Response evaluation system
├── linkedIn_post/               # Content generation
├── sources/                     # Source PDF and video files
├── chunked/                     # Processed document chunks  
├── embed_db/                    # ChromaDB vector database
├── video_transcripts/           # Extracted video transcripts
├── source_to_text/              # Text extraction modules
├── chunking/                    # Document chunking logic
├── embed/                       # Vector embedding functionality
├── retrieve/                    # Document retrieval logic
└── utils/                       # Utility functions and logging
```

## Advanced Features

### 🔍 Semantic Search Strategy
- **Local Focus**: RAG architecture, LangChain, LangGraph, vector databases
- **Web Focus**: Current events (2024-2026), news, stock prices, general knowledge
- **Intelligent Routing**: LLM-powered decision making with reasoning

### 📝 Structured Outputs
- **SearchStrategy**: Decision and reasoning for information source selection
- **DocumentRelevance**: Confidence scoring for retrieved documents  
- **QueryRewrite**: Improved query formulations for better retrieval
- **LinkedInPost**: Structured social media content with hashtags and CTAs

### 🔄 Self-Improving Workflow
- **Reflection Loops**: Automatic query refinement when results are insufficient
- **Fallback Mechanisms**: Web search when local documents are inadequate  
- **Loop Prevention**: Smart iteration limits to prevent infinite loops
- **Quality Gates**: Relevance thresholds for content acceptance

## Help

Use the `-h` or `--help` flag to display usage instructions:

```bash
python3 main.py -h
```

This displays all available commands including agentic queries and LinkedIn post generation options.

## Example Workflows

### Technical Query (Local Documents)
```bash
python3 main.py "Why is hybrid search better than vector-only search"
# → Semantic routing chooses local docs
# → Retrieves relevant chunks about hybrid search techniques
# → Generates comprehensive technical answer
# → Evaluates response quality automatically
```

### Current Events Query (Web Search)  
```bash
python3 main.py "What are the latest AI developments in 2026?"
# → Semantic routing chooses web search
# → Searches DuckDuckGo for current information
# → Synthesizes real-time results
# → Provides up-to-date information with evaluation
```

### Content Creation
```bash
python3 main.py --linkedin_self_analyze
# → Agent analyzes its own capabilities
# → Generates professional LinkedIn post
# → Includes technical achievements and learning journey
# → Ready for social media publication
```

