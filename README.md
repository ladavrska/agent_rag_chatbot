# Document Retrieval Pipeline

A complete RAG (Retrieval-Augmented Generation) pipeline that processes PDFs and video transcripts, chunks them, embeds them into a vector database, and provides retrieval functionality.

## Features

- **PDF Processing**: Extract text from PDF files
- **Video Processing**: Extract transcripts from video files
- **Text Chunking**: Split documents into manageable chunks using recursive chunking
- **Vector Embeddings**: Create and store embeddings in a ChromaDB vector database
- **Retrieval**: Query the vector database to find relevant document chunks
- **Smart Pipeline**: Automatically detects first run vs. subsequent runs

## Usage

### Command Line Usage

```bash
# Query with quoted string
python3 main.py "What is RAG?"

# Query with multiple words (no quotes needed)
python3 main.py What is retrieval augmented generation

# Interactive mode (no query provided)
python3 main.py

# Show help
python3 main.py -h
python3 main.py --help
```

### Pipeline Behavior

#### First Run
When you run the script for the first time (no existing vector database or chunked files), it will automatically execute the full pipeline:

1. **PDF to Text**: Extract text from all PDF files matching the configured pattern
2. **Video to Text**: Extract transcripts from video files
3. **Chunking**: Split transcripts into chunks using recursive chunking
4. **Embedding**: Create vector embeddings and store in ChromaDB
5. **Retrieval**: Query the database with your provided query or enter interactive mode

#### Subsequent Runs
After the initial setup, the script will only run the retrieval step, querying the existing vector database.

### Interactive Mode

If you don't provide a query, the script enters interactive mode where you can:
- Enter queries interactively
- Test different search terms
- Explore the embedded documents

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

The pipeline uses configuration settings from `config.py` for:
- PDF file patterns
- Video file patterns
- Output directories
- Database persistence directory

## Project Structure

```
├── main.py                    # Main pipeline orchestration
├── config.py                  # Configuration settings
├── requirements.txt           # Python dependencies
├── sources/                   # Source PDF and video files
├── chunked/                   # Processed and chunked documents
├── embed_db/                  # ChromaDB vector database
├── video_transcripts/         # Extracted video transcripts
├── source_to_text/           # PDF and video text extraction
├── chunking/                 # Document chunking logic
├── embed/                    # Vector embedding functionality
├── retrieve/                 # Document retrieval logic
└── utils/                    # Utility functions and logging
```

## Help

Use the `-h` or `--help` flag to display usage instructions:

```bash
python3 main.py -h
```

This will show the basic usage patterns and exit without running the pipeline.

