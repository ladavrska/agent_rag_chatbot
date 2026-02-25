# Agentic RAG Chatbot

An intelligent AI assistant that combines local document knowledge with real-time web search. The system automatically decides whether to search your documents or the internet based on your question type.

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Basic Usage
```bash
# Ask technical questions (searches local documents)
python3 main.py "What is LangGraph?"

# Ask current events (searches the web)  
python3 main.py "What are the latest AI developments in 2026?"

# Generate LinkedIn posts
python3 main.py --linkedin_self_analyze

# Get help
python3 main.py --help
```

## Features

- **Smart Search Routing**: Automatically chooses between local documents and web search
- **Quality Evaluation**: AI judges response accuracy, relevance, and clarity  
- **LinkedIn Content**: Generate professional posts with AI self-analysis
- **Multi-format Processing**: PDF and video transcript support

## How It Works

### First Time Setup
On your first run, the system will automatically:
1. Extract text from your PDF and video files
2. Process and chunk the content  
3. Create a searchable vector database
4. Run your first query

### Automatic Smart Routing
- **Technical Questions** → Searches your local documents
- **Current Events** → Searches the web in real-time
- **Poor Results** → Automatically refines the query and tries again

## Command Options

```bash
# Basic queries
python3 main.py "Your question here"

# LinkedIn content generation  
python3 main.py --linkedin           # Standard post
python3 main.py --linkedin_self_analyze  # AI self-analysis post
python3 main.py -lsa                 # Short form

# Help
python3 main.py --help
```

## Example Queries

### Technical (Local Documents)
```bash
python3 main.py "Why is hybrid search better than vector-only search"
# ✅ Searches your technical documents
# ✅ Provides detailed technical explanation  
# ✅ Evaluates response quality automatically
```

### Current Events (Web Search)
```bash
python3 main.py "What are the latest AI developments in 2026?"
# ✅ Searches the web for current information
# ✅ Provides up-to-date results
# ✅ Combines multiple sources
```

## Configuration

Place your source files in:
- `sources/` - PDF documents and video files
- Configuration options available in `config.py`

## Technical Details

For detailed architecture, technology stack, and implementation details, see [architecture.mmd](architecture.mmd).

## Need Help?

```bash
python3 main.py --help
```

