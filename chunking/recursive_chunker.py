import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import (VIDEO_TRANSCRIPT_DIR, CHUNKED_DIR, CHUNK_SIZE, CHUNK_OVERLAP)
from utils.chunking_utils import load_transcript_documents, validate_loaded_documents, split_documents_into_chunks, documents_to_dicts, save_chunks_to_json
from utils.log_utils import print_chunking_summary


def chunk_recursive(input_dir:str, output_dir:str):

    # 1. Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP, 
        add_start_index=True
    )

    # 2. Setup paths and lists
    transcript_folder = Path(input_dir)
    output_folder = Path(output_dir)
    output_folder.mkdir(exist_ok=True)

    # Check if transcript folder exists
    if not transcript_folder.exists():
        print(f"Error: {transcript_folder} folder not found!")
        return

    # 3. Load transcript documents
    documents = load_transcript_documents(transcript_folder)
    
    # Validate loaded documents
    if not validate_loaded_documents(documents):
        return

    # 4. Split all documents into chunks
    all_splits = split_documents_into_chunks(text_splitter, documents)

    # 5. Show summary of chunking results
    print_chunking_summary(all_splits)
    
    # 6. Convert Document objects to dicts for saving
    docs_to_save = documents_to_dicts(all_splits)

    # 7. Save chunks to JSON file
    save_chunks_to_json(docs_to_save, output_folder)


def main():
    print("Starting transcript chunking process...")
    chunk_recursive(input_dir=VIDEO_TRANSCRIPT_DIR, output_dir=CHUNKED_DIR)

if __name__ == "__main__":
    main()   
