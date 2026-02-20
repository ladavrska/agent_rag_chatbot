import json
import glob
from pathlib import Path

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings



def load_chunk_files(pattern):
    """Load and process multiple chunk files"""
    chunk_files = glob.glob(pattern)
    
    if not chunk_files:
        raise FileNotFoundError(f"No files found matching pattern: {pattern}")
    
    print(f"Found {len(chunk_files)} chunk files: {[Path(f).name for f in chunk_files]}")
    
    all_documents = []
    
    for chunk_file in chunk_files:
        print(f"Processing: {chunk_file}")
            
        # Load and validate JSON structure
        try:
            with open(chunk_file, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            
            if not json_data:
                print(f"Warning: {chunk_file} is empty, skipping...")
                continue
                
            # Validate required keys
            file_documents = []
            for item in json_data:
                if "page_content" not in item or "metadata" not in item:
                    print(f"Warning: Invalid JSON structure in {chunk_file}, missing required keys")
                    continue
                
                # Add source file information to metadata
                enhanced_metadata = item["metadata"].copy()

                enhanced_metadata["source_file"] = Path(chunk_file).name
                enhanced_metadata["source_path"] = chunk_file
                enhanced_metadata["original_length"] = len(item["page_content"])
                
                file_documents.append(
                    Document(page_content=item["page_content"], metadata=enhanced_metadata)
                )
            
            all_documents.extend(file_documents)
            print(f"  → Loaded {len(file_documents)} chunks from {chunk_file}")
                
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format in {chunk_file}: {e}")
            continue
        except Exception as e:
            print(f"Error processing {chunk_file}: {e}")
            continue
    
    if not all_documents:
        raise ValueError("No valid documents were loaded from any chunk files")
    
    print(f"\nTotal loaded: {len(all_documents)} chunks from {len(chunk_files)} files")
    return all_documents
    

def initialize_ollama_embeddings(model_name: str) -> OllamaEmbeddings:
    """Initialize and test Ollama embeddings connection"""
    try:
        # Basic configuration (local Ollama running on default port)
        embeddings = OllamaEmbeddings(model=model_name)
        
        # Test Ollama connection
        print("Testing Ollama connection...")
        embeddings.embed_query("connection test")
        print("✓ Ollama connection successful")
        return embeddings
    except Exception as e:
        raise ConnectionError(f"Cannot connect to Ollama: {e}")