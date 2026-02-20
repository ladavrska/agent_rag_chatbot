from pathlib import Path
from langchain_core.documents import Document


def print_header(message: str) -> None:
    """Print a formatted header with equals signs border"""
    print(f"\n{'='*60}")
    print(f"{message}")
    print(f"{'='*60}")


def log_processing_count(count:int) -> None:
    """Log the start of PDF processing"""
    print(f"Found {count} PDF files to process:")


def log_processing_file(file: str) -> None:
    """Log processing of individual file"""
    print(f"\nProcessing: {file}")


def log_extraction_summary(successful_extractions: list[tuple[str, str, int]], 
                          failed_extractions: list[str], 
                          total_documents: int, 
                          output_dir: str
) -> None:
    """Log the complete extraction summary"""
    print_header("Extraction Summary")
    
    print(f"Successfully processed: {len(successful_extractions)} PDFs")
    print(f"Failed: {len(failed_extractions)} PDFs")
    print(f"Total documents extracted: {total_documents:,}")
    print(f"Output directory: {output_dir}")
    
    if successful_extractions:
        print(f"\n📄 Successfully extracted files:")
        for pdf_file, pdf_path, doc_count in successful_extractions:
            print(f"  {Path(pdf_file).name} -> {Path(pdf_path).name} ({doc_count:,} documents/chunks)")
    
    if failed_extractions:
        print(f"\n❌ Failed extractions:")
        for pdf_path in failed_extractions:
            print(f"  {Path(pdf_path).name}")
            
            
def print_chunking_summary(all_splits: list[Document]) -> None:
    """Print summary of chunking results by source file"""
    
    print_header("CHUNKING SUMMARY")
    
    source_summary = {}
    for chunk in all_splits:
        source = chunk.metadata.get("source", "unknown")
        source_summary[source] = source_summary.get(source, 0) + 1

    for source, count in source_summary.items():
        print(f"  {source}: {count} chunks")
        
        
def log_embedding_summary(documents: list[Document]) -> None:
    """Log summary of embedding results by source file"""
    
    print(f"✓ Vector DB created with {len(documents)} chunks from multiple files")
    
    print_header("EMBEDDING SUMMARY")
    
    source_summary = {}
    for doc in documents:
        source = doc.metadata.get("source_file", "unknown")
        source_summary[source] = source_summary.get(source, 0) + 1

    print("\nDocument distribution by source file:")
    for source, count in source_summary.items():
        print(f"  {source}: {count} chunks")


def print_usage():
    """Print usage instructions"""
    print("Usage:")
    print("  python3 main.py \"What is RAG?\"")
    print("  python3 main.py What is retrieval augmented generation")
    print("  python3 main.py  # Interactive mode")