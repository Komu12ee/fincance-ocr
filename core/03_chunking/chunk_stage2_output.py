"""
Chunk Stage 2 OCR output using Docling's HybridChunker.

This script takes the merged structured.md from stage2_output and chunks it
using semantic chunking (token-based, respecting document structure).

Pipeline Overview:
  Stage 1: PDF → preprocessed images (PNG)
  Stage 2: Images → Docling OCR + confidence scoring
           If confidence < 0.6 → Sarvam AI fallback
           Merge all pages into single structured.md/json
  Stage 3: structured.md → semantic chunks (THIS SCRIPT)
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List
import json

try:
    from docling.document_converter import DocumentConverter
    from docling.chunking import HybridChunker
    from transformers import AutoTokenizer
except ImportError:
    print("Error: Required packages not installed.")
    print("Run: pip install docling transformers torch")
    sys.exit(1)

# Add core/ to path for shared pipeline config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pipeline_config as pc

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class Stage2Chunker:
    """Chunk Stage 2 structured.md output into semantic chunks."""
    
    def __init__(self, model: str = "BAAI/bge-m3", max_tokens: int = 1024, merge_peers: bool = True):
        """
        Initialize chunker.
        
        Args:
            model: Embedding model for tokenization
            max_tokens: Maximum tokens per chunk
            merge_peers: Whether to merge sibling sections
        """
        self.model = model
        self.max_tokens = max_tokens
        self.merge_peers = merge_peers
        
        logger.info(f"Loading tokenizer: {model}")
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        
        logger.info("Loading Docling converter")
        self.converter = DocumentConverter()
        
        logger.info(f"Initializing HybridChunker (max_tokens={max_tokens}, merge_peers={merge_peers})")
        self.chunker = HybridChunker(
            tokenizer=self.tokenizer,
            max_tokens=max_tokens,
            merge_peers=merge_peers
        )
    
    def load_confidence_data(self, confidence_json: Path) -> Dict[int, dict]:
        """Load confidence scores for each page."""
        try:
            with open(confidence_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {p['page_number']: p for p in data.get('pages', [])}
        except Exception as e:
            logger.warning(f"Could not load confidence data: {e}")
            return {}
    
    def _extract_pages_from_chunk(self, chunk_text: str) -> list:
        """Extract page numbers from <!-- Page X --> comments in chunk text."""
        import re
        pages = []
        for match in re.finditer(r'<!-- Page (\d+) -->', chunk_text):
            page_num = int(match.group(1))
            if page_num not in pages:
                pages.append(page_num)
        return sorted(pages)
    
    def _get_page_boundaries(self, markdown_file: Path) -> dict:
        """
        Extract page boundaries from raw markdown.
        Returns dict: {page_num: character_position}
        """
        import re
        boundaries = {}
        with open(markdown_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        for match in re.finditer(r'<!-- Page (\d+) -->', content):
            page_num = int(match.group(1))
            boundaries[page_num] = match.start()
        
        return boundaries
    
    def _find_pages_in_chunk(self, chunk_text: str, page_boundaries: dict, 
                            chunk_index: int, total_chunks: int) -> str:
        """
        Estimate which pages are in this chunk based on text content and page boundaries.
        This is a heuristic since we don't have exact position tracking through Docling conversion.
        """
        # Extract first non-whitespace content snippet
        snippet = chunk_text.strip()[:100].lower()
        
        # If we have page boundaries, try to estimate based on chunk position
        if not page_boundaries:
            return "Unknown"
        
        pages = sorted(page_boundaries.keys())
        # Rough estimation: distribute pages across chunks
        pages_per_chunk = len(pages) / total_chunks
        start_idx = int(chunk_index * pages_per_chunk)
        end_idx = int((chunk_index + 1) * pages_per_chunk)
        
        estimated_pages = pages[max(0, start_idx):min(len(pages), end_idx + 1)]
        
        if estimated_pages:
            if len(estimated_pages) == 1:
                return str(estimated_pages[0])
            else:
                return f"{estimated_pages[0]}-{estimated_pages[-1]}"
        
        return "Unknown"
    
    def _format_page_range(self, pages: list) -> str:
        """Format page numbers as range (e.g., 1-3 or 1,3,5)."""
        if not pages:
            return "Unknown"
        if len(pages) == 1:
            return str(pages[0])
        
        # Check if consecutive
        if pages[-1] - pages[0] + 1 == len(pages):
            return f"{pages[0]}-{pages[-1]}"
        else:
            return ",".join(map(str, pages))

    def chunk_document(self, input_md: Path, output_dir: Path, confidence_json: Path = None, pdf_name: str = None) -> int:
        """
        Chunk a single structured.md file.
        
        Args:
            input_md: Path to structured.md
            output_dir: Directory to save chunks
            confidence_json: Path to confidence JSON for metadata
            pdf_name: Original PDF filename for metadata
            
        Returns:
            Number of chunks created
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        doc_name = input_md.stem
        
        logger.info(f"Processing: {input_md.name}")
        
        # Load confidence data for metadata
        page_confidence = {}
        if confidence_json and confidence_json.exists():
            page_confidence = self.load_confidence_data(confidence_json)
            logger.info(f"  Loaded confidence for {len(page_confidence)} pages")
        
        # Extract PDF name if not provided
        if not pdf_name and confidence_json:
            pdf_name = confidence_json.stem.replace('.pdf_confidence', '')
        elif not pdf_name:
            pdf_name = doc_name
        
        try:
            # Convert markdown to Docling document
            logger.info(f"  Converting to Docling document...")
            result = self.converter.convert(str(input_md))
            doc = result.document
            
            # Chunk the document
            logger.info(f"  Chunking document (max_tokens={self.max_tokens})...")
            chunks = list(self.chunker.chunk(doc))
            logger.info(f"  Generated {len(chunks)} chunks")
            
            # Get page boundaries from raw markdown
            page_boundaries = self._get_page_boundaries(input_md)
            logger.info(f"  Found {len(page_boundaries)} pages in document")
            
            # Save chunks
            total_tokens = 0
            for i, chunk in enumerate(chunks, 1):
                chunk_num = f"{i:03d}"
                filename = f"{doc_name}_chunk_{chunk_num}.txt"
                filepath = output_dir / filename
                
                chunk_text = chunk.text
                token_count = len(self.tokenizer.encode(chunk_text))
                total_tokens += token_count
                
                # Estimate which pages are in this chunk
                page_range = self._find_pages_in_chunk(chunk_text, page_boundaries, i-1, len(chunks))
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    # Write metadata
                    f.write(f"# Chunk {i}\n")
                    f.write(f"Document: {pdf_name}\n")
                    f.write(f"Pages: {page_range}\n")
                    f.write(f"Tokens: {token_count}\n")
                    
                    # Write document structure metadata
                    if hasattr(chunk.meta, 'headings') and chunk.meta.headings:
                        f.write(f"Headings: {chunk.meta.headings}\n")
                    
                    f.write(f"\n---\n\n")
                    f.write(chunk_text)
                    f.write("\n")
            
            logger.info(f"✅ {doc_name}: {len(chunks)} chunks, {total_tokens} total tokens")
            logger.info(f"   Saved to: {output_dir}")
            
            # Save chunk metadata
            metadata = {
                "document": input_md.name,
                "total_chunks": len(chunks),
                "total_tokens": total_tokens,
                "avg_tokens_per_chunk": round(total_tokens / len(chunks)) if chunks else 0,
                "model": self.model,
                "max_tokens_per_chunk": self.max_tokens,
                "chunks": []
            }
            
            for i in range(1, len(chunks) + 1):
                chunk_file = f"{doc_name}_chunk_{i:03d}.txt"
                metadata["chunks"].append({
                    "index": i,
                    "filename": chunk_file
                })
            
            metadata_path = output_dir / f"{doc_name}_chunks_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"   Metadata: {metadata_path}")
            
            return len(chunks)
            
        except Exception as e:
            logger.error(f"❌ Error chunking {input_md}: {e}")
            import traceback
            traceback.print_exc()
            return 0


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Chunk Stage 2 OCR output using semantic chunking"
    )
    parser.add_argument('--input', '-i', type=str,
                       help='Input stage2_output folder or specific structured.md')
    parser.add_argument('--output', '-o', type=str,
                       help='Output directory for chunks')
    parser.add_argument('--document', '-d', type=str, default=None,
                       help='Specific document folder to chunk (if not specified, chunks all documents)')
    parser.add_argument('--model', '-m', type=str, default='BAAI/bge-m3',
                       help='Embedding model (default: BAAI/bge-m3)')
    parser.add_argument('--max-tokens', type=int, default=1024,
                       help='Maximum tokens per chunk (default: 1024)')
    
    args = parser.parse_args()
    
    # Paths (pipeline-aware: data/<pipeline>/stage2 -> data/<pipeline>/chunks)
    if args.input:
        input_dir = Path(args.input)
    else:
        input_dir = pc.paths.stage2

    output_dir = Path(args.output) if args.output else pc.paths.chunks
    
    logger.info("=" * 80)
    logger.info("Stage 2 → Stage 3: Document Chunking")
    logger.info("=" * 80)
    logger.info(f"Input:  {input_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Model:  {args.model}")
    logger.info(f"Max tokens per chunk: {args.max_tokens}")
    logger.info("=" * 80)
    
    # Initialize chunker
    chunker = Stage2Chunker(
        model=args.model,
        max_tokens=args.max_tokens
    )
    
    # Process documents
    total_chunks = 0
    
    if args.document:
        # Process specific document
        doc_path = input_dir / args.document
        
        if not doc_path.exists():
            logger.error(f"Document folder not found: {doc_path}")
            logger.info(f"Available documents:")
            for doc_dir in input_dir.iterdir():
                if doc_dir.is_dir() and (doc_dir / 'structured.md').exists():
                    logger.info(f"  - {doc_dir.name}")
            return
        
        structured_md = doc_path / 'structured.md'
        confidence_json = doc_path / f"{args.document}.pdf_confidence.json"
        
        if not structured_md.exists():
            logger.error(f"structured.md not found: {structured_md}")
            return
        
        # Extract PDF name from confidence file
        pdf_name = args.document
        if confidence_json.exists():
            pdf_name = confidence_json.stem.replace('.pdf_confidence', '')
        
        chunks = chunker.chunk_document(
            structured_md,
            output_dir / args.document,
            confidence_json if confidence_json.exists() else None,
            pdf_name=pdf_name
        )
        total_chunks += chunks
    else:
        # Process all documents in stage2_output
        for doc_dir in sorted(input_dir.iterdir()):
            if not doc_dir.is_dir():
                continue
            
            structured_md = doc_dir / 'structured.md'
            if not structured_md.exists():
                continue
            
            confidence_json = doc_dir / f"{doc_dir.name}.pdf_confidence.json"
            
            # Extract PDF name from confidence file
            pdf_name = doc_dir.name
            if confidence_json.exists():
                pdf_name = confidence_json.stem.replace('.pdf_confidence', '')
            
            chunks = chunker.chunk_document(
                structured_md,
                output_dir / doc_dir.name,
                confidence_json if confidence_json.exists() else None,
                pdf_name=pdf_name
            )
            total_chunks += chunks
    
    logger.info("=" * 80)
    logger.info(f"✅ Chunking complete! Total chunks: {total_chunks}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
