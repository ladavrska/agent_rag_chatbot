import json
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNKED_TRANSCRIPTS_FILE

def load_transcript_documents(transcript_folder: Path) -> list[Document]:
    """Load and process transcript files into Document objects"""
    documents = []
    
    # Loop through every .txt file in the folder
    for file_path in transcript_folder.glob("*.txt"):
        print(f"Processing {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip() 
            # Skip empty files
            if not content:
                print(f"  Warning: {file_path.name} is empty, skipping...")
                continue
            
            # Create a Document object with the full text and metadata
            doc = Document(
                page_content=content,
                metadata={
                    "source": file_path.name,
                    "category": "video_transcript",
                    "file_path": str(file_path),
                    "original_length": len(content)
                }
            )
            documents.append(doc)
            print(f"  ✓ Loaded {len(content)} characters from {file_path.name}")
            
        except Exception as e:
            print(f"  ❌ Error reading {file_path.name}: {e}")
            continue
    
    return documents


def validate_loaded_documents(documents: list[Document]) -> bool:
    """Validate that documents were successfully loaded"""
    if not documents:
        print("No valid documents found to process!")
        return False
    
    print(f"\nLoaded {len(documents)} transcript files")
    return True


def split_documents_into_chunks(
    text_splitter: RecursiveCharacterTextSplitter, 
    documents: list[Document]
) -> list[Document]:
    """Split documents into chunks using the provided text splitter"""
    print("Splitting documents into chunks...")
    all_splits = text_splitter.split_documents(documents)
    
    print(f"Created {len(all_splits)} chunks from {len(documents)} transcript files")
    return all_splits


def documents_to_dicts(documents: list[Document]) -> list[dict]:
    """Convert Document objects to list of dictionaries"""
    return [
        {
            "page_content": doc.page_content, 
            "metadata": doc.metadata
        } 
        for doc in documents
    ]
    
    
def save_chunks_to_json(docs_to_save: list[dict], output_folder: Path, filename: str = CHUNKED_TRANSCRIPTS_FILE) -> None:
    """Save chunked transcripts to JSON file"""
    
    output_file = output_folder / filename
    
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(docs_to_save, f, ensure_ascii=False, indent=4)
        
        print(f"\n✓ Successfully saved {len(docs_to_save)} chunks to {output_file}")
    except Exception as e:
        print(f"❌ Error saving to {output_file}: {e}")