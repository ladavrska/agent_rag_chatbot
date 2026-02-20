import os
import glob
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from utils.log_utils import print_header, log_processing_count, log_processing_file, log_extraction_summary
from utils.source_to_text_utils import save_to_json
from config import (PDF_PATTERN, CHUNKED_DIR)


def pdf_to_json(pdf_path:str, output_dir:str):
    """
    This script creates chunked PDF JSON files directly from PDFs.
    1. Load PDF using PyPDFLoader
    2. Convert Document objects to dicts
    3. Save as JSON file
    4. Return output path and number of documents extracted
    """
    
    output_data = []
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load() # documents: [Document, Document, ...]
        
        print_header(f"✓ Loaded: {pdf_path} with {len(documents)} pages." )

        documents = [
            {"page_content": doc.page_content, "metadata": doc.metadata} 
            for doc in documents
        ]
        
        output_data.extend(documents)
    except Exception as e:
        print(f"✗ Error processing {pdf_path}: {str(e)}")
        return None, 0

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
        
    # Generate output filename based on input PDF name
    pdf_name = Path(pdf_path).stem
    output_filename = f"chunked_{pdf_name}.json"
    output_path = os.path.join(output_dir, output_filename)

    # Save to a file
    save_to_json(output_data, output_path)
        
    print(f"✓ Extracted: {pdf_path} -> {output_path}")
    return output_path, len(documents)


def extract_text_from_pdfs(pdf_pattern:str, output_dir:str):
    """Extract multiple PDFs to JSON files"""
    
    pdf_files = glob.glob(pdf_pattern)
    
    if not pdf_files:
        print(f"No PDF files found matching pattern: {pdf_pattern}")
        return
    
    successful_extractions = []
    failed_extractions = []
    total_documents = 0
    
    log_processing_count(len(pdf_files))
    
    for pdf_file in pdf_files:
        log_processing_file(pdf_file)
        result_path, doc_count = pdf_to_json(pdf_file, output_dir)
        
        if result_path:
            successful_extractions.append((pdf_file, result_path, doc_count))
            total_documents += doc_count
        else:
            failed_extractions.append(pdf_file)
    
    log_extraction_summary(successful_extractions, failed_extractions, total_documents, output_dir)
    
    return successful_extractions, failed_extractions



if __name__ == "__main__":
    """Extract PDFs to Text files"""
    extract_text_from_pdfs(PDF_PATTERN, CHUNKED_DIR)
    
