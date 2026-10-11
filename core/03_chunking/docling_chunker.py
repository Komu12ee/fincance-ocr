"""
Stage 3 - Intelligent Document Chunking
Converts optimized documents into semantic chunks using Docling.

Usage:
    python docling_chunker.py                              # Use default directories
    python docling_chunker.py --input /path/to/docs      # Specify input folder
    python docling_chunker.py --input file.md --output ./chunks  # Single file
    python docling_chunker.py --config config.json       # Load config from file
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Dict
import os
import json

try:
    from docling.document_converter import DocumentConverter
    from docling.chunking import HybridChunker
    from transformers import AutoTokenizer
except ImportError:
    print("Error: Required packages not installed. Run: pip install -r requirements.txt")
    sys.exit(1)

# Add parent directory to path to import config_manager
sys.path.insert(0, str(Path(__file__).parent.parent))
from config_manager import Config, PathManager


logger = logging.getLogger(__name__)


class DoclingChunker:
    """Intelligent document chunker using Docling."""
    
    def __init__(self, model: str = "BAAI/bge-m3", max_tokens: int = 1024, merge_peers: bool = True):
        """
        Initialize chunker.
        
        Args:
            model: Embedding model to use
            max_tokens: Maximum tokens per chunk
            merge_peers: Whether to merge peer sections
        """
        self.model = model
        self.max_tokens = max_tokens
        self.merge_peers = merge_peers
        
        logger.info(f"Loading model: {model}")
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.converter = DocumentConverter()
        self.chunker = HybridChunker(
            tokenizer=self.tokenizer,
            max_tokens=max_tokens,
            merge_peers=merge_peers
        )
    
    def chunk_file(self, input_path: str, output_dir: str, file_mapping: Optional[Dict] = None) -> int:
        """
        Chunk a single document.
        
        Args:
            input_path: Path to input document (md, pdf, docx)
            output_dir: Directory to save chunks
            file_mapping: Optional mapping of output names to original names
            
        Returns:
            Number of chunks created
        """
        try:
            input_path = Path(input_path)
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Processing: {input_path.name}")
            
            result = self.converter.convert(str(input_path))
            doc = result.document
            chunks = list(self.chunker.chunk(doc))
            
            doc_name = input_path.stem  # Filename without extension
            original_name = file_mapping.get(doc_name, doc_name) if file_mapping else doc_name
            
            for i, chunk in enumerate(chunks, 1):
                filename = f"{doc_name}_chunk_{i:03d}.txt"
                filepath = output_dir / filename
                
                with open(filepath, "w", encoding="utf-8") as f:
                    if hasattr(chunk.meta, 'headings') and chunk.meta.headings:
                        f.write(f"Headings: {chunk.meta.headings}\n")
                    f.write(f"Source: {original_name}\n")
                    f.write(f"---\n")
                    f.write(f"{chunk.text}\n")
            
            logger.info(f"✅ {doc_name}: saved {len(chunks)} chunks")
            return len(chunks)
            
        except Exception as e:
            logger.error(f"❌ Error processing {input_path}: {e}")
            return 0
    
    def chunk_batch(self, input_dir: str, output_dir: str, pattern: str = "*.md", 
                   file_mapping: Optional[Dict] = None) -> Dict[str, int]:
        """
        Chunk all documents in a directory.
        
        Args:
            input_dir: Directory containing documents
            output_dir: Directory to save chunks
            pattern: File pattern to match (e.g., "*.md")
            file_mapping: Optional mapping of output names to original names
            
        Returns:
            Dictionary with filename: chunk_count
        """
        input_dir = Path(input_dir)
        results = {}
        
        if not input_dir.exists():
            logger.error(f"Input directory not found: {input_dir}")
            return results
        
        files = sorted(input_dir.glob(pattern))
        logger.info(f"Found {len(files)} files matching {pattern}")
        
        for doc_path in files:
            chunk_count = self.chunk_file(doc_path, output_dir, file_mapping)
            results[doc_path.name] = chunk_count
        
        return results


def parse_file_mapping(mapping_file: str) -> Dict[str, str]:
    """
    Parse mapping file to create mapping from output names to original names.
    
    Expected format: "Original Name.pdf -------->output_corrected1.pdf"
    
    Args:
        mapping_file: Path to mapping file
        
    Returns:
        Dictionary mapping document names
    """
    mapping = {}
    
    try:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '-------->' in line:
                    parts = [p.strip() for p in line.split('-------->')]
                    if len(parts) == 2:
                        original_name = parts[0].strip()
                        output_name = os.path.splitext(parts[1].strip())[0]
                        mapping[output_name] = original_name
    except FileNotFoundError:
        logger.warning(f"Mapping file not found: {mapping_file}")
    
    return mapping


def main():
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)-7s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    parser = argparse.ArgumentParser(
        description="Stage 3 - Intelligent Document Chunking with Docling"
    )
    parser.add_argument('--input', '-i', type=str,
                       help='Input directory or file')
    parser.add_argument('--output', '-o', type=str,
                       help='Output directory for chunks')
    parser.add_argument('--pattern', '-p', type=str, default='*.md',
                       help='File pattern to match (default: *.md)')
    parser.add_argument('--model', '-m', type=str, default='BAAI/bge-m3',
                       help='Embedding model (default: BAAI/bge-m3)')
    parser.add_argument('--max-tokens', type=int, default=1024,
                       help='Maximum tokens per chunk (default: 1024)')
    parser.add_argument('--mapping', type=str,
                       help='Path to file mapping file (e.g., files.txt)')
    parser.add_argument('--config', '-c', type=str,
                       help='Config file (JSON)')
    parser.add_argument('--show-config', action='store_true',
                       help='Print configuration and exit')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config(stage='chunking')
    
    # Override with config file if provided
    if args.config:
        file_config = Config.load_from_file(args.config)
        config.config_dict.update(file_config)
    
    # Override with command-line arguments
    if args.input:
        config.config_dict['input_dir'] = args.input
    if args.output:
        config.config_dict['output_dir'] = args.output
    
    # Show configuration if requested
    if args.show_config:
        config.log_config()
        return
    
    # Get paths
    input_path_str = config.get_input_path(as_str=True)
    output_path_str = config.get_output_path(as_str=True)
    
    logger.info(f"Input: {input_path_str}")
    logger.info(f"Output: {output_path_str}")
    
    # Create output directory
    PathManager.ensure_dirs(output_path_str)
    
    # Load file mapping if provided
    file_mapping = None
    if args.mapping:
        file_mapping = parse_file_mapping(args.mapping)
        logger.info(f"Loaded mapping for {len(file_mapping)} files")
    
    # Initialize chunker
    chunker = DoclingChunker(
        model=args.model,
        max_tokens=args.max_tokens
    )
    
    # Process files
    input_path = Path(input_path_str)
    
    if input_path.is_file():
        # Single file
        total_chunks = chunker.chunk_file(
            str(input_path),
            output_path_str,
            file_mapping
        )
        logger.info(f"\n✅ Complete! Total chunks: {total_chunks}")
    else:
        # Directory
        results = chunker.chunk_batch(
            input_path_str,
            output_path_str,
            pattern=args.pattern,
            file_mapping=file_mapping
        )
        
        total_chunks = sum(results.values())
        logger.info(f"\n✅ Complete! Processed {len(results)} files, {total_chunks} total chunks")
        
        # Print summary
        for filename, count in results.items():
            logger.info(f"  {filename}: {count} chunks")


if __name__ == '__main__':
    main()
