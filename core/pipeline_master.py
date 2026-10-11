#!/usr/bin/env python3
"""
Production-Ready RAG Pipeline orchestrator (shared core).
Orchestrates: PDF Input -> Stage 1 -> Stage 2 -> Stage 3 -> Stage 4 (Embeddings)
for a selected pipeline (chips / fg / fg2), reading/writing data/<pipeline>/...

Usage:
    python pipeline_master.py --pipeline chips            # Run full pipeline
    python pipeline_master.py --pipeline fg --stage 2     # Run from Stage 2 onwards
    python pipeline_master.py --dry-run                   # Show what would run
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent  # = core/

# Use the interpreter running this orchestrator (the project's uv venv),
# so stages run in the same environment without a hardcoded venv path.
PY = Path(sys.executable)

# Stage scripts
STAGE1_SCRIPT = PROJECT_ROOT / "01_preprocessing" / "run_stage1.py"
STAGE2_SCRIPT = PROJECT_ROOT / "01_preprocessing" / "run_stage2.py"
STAGE3_SCRIPT = PROJECT_ROOT / "03_chunking" / "chunk_stage2_output.py"
STAGE4_SCRIPT = PROJECT_ROOT / "04_embeddings_and_kg" / "scripts" / "embeddings_production.py"


def check_input_pdfs(input_dir: Path) -> bool:
    """Check for PDFs in the pipeline's input folder."""
    if not input_dir.exists():
        logger.error(f"Input folder not found: {input_dir}")
        return False
    pdfs = list(input_dir.glob("*.pdf"))
    if not pdfs:
        logger.warning(f"No PDFs found in {input_dir}")
        return False
    logger.info(f"Found {len(pdfs)} PDF(s)")
    for pdf in pdfs:
        logger.info(f"  - {pdf.name}")
    return True


def run_stage(stage_num, script_path, env):
    """Run a pipeline stage as a subprocess (inherits PIPELINE via env)."""
    if not script_path.exists():
        logger.error(f"Script not found: {script_path}")
        return False

    logger.info("=" * 70)
    logger.info(f"Stage {stage_num}: {script_path.name}")
    logger.info("=" * 70)

    try:
        subprocess.run([str(PY), str(script_path)], check=True, env=env)
        logger.info(f"Stage {stage_num} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Stage {stage_num} failed with exit code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline Orchestrator (shared core)")
    parser.add_argument('--pipeline', type=str, default=os.getenv("PIPELINE", "chips"),
                        help='Pipeline to run: chips | fg | fg2 (default: chips or $PIPELINE)')
    parser.add_argument('--stage', type=int, default=1,
                        help='Start from stage (1-4, default: 1)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be processed without running')
    parser.add_argument('--skip-embeddings', action='store_true',
                        help='Skip Stage 4 (embeddings)')
    args = parser.parse_args()

    # Select the pipeline for this process and all spawned stages.
    os.environ["PIPELINE"] = args.pipeline.strip().lower()
    sys.path.insert(0, str(PROJECT_ROOT))
    import pipeline_config as pc  # imported after PIPELINE is set
    pc.ensure_dirs()
    env = dict(os.environ)

    logger.info("=" * 70)
    logger.info(f"RAG Pipeline Orchestrator  |  pipeline={pc.PIPELINE}")
    logger.info(f"Data root: {pc.DATA_ROOT}")
    logger.info("=" * 70)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if not check_input_pdfs(pc.paths.input_pdfs):
        if args.dry_run:
            logger.warning("(Dry-run: skipping input validation)")
        else:
            logger.error("No PDFs to process. Exiting.")
            return 1

    if args.dry_run:
        logger.info("DRY-RUN MODE: Would execute:")
        if args.stage <= 1: logger.info("  -> Stage 1: run_stage1.py")
        if args.stage <= 2: logger.info("  -> Stage 2: run_stage2.py")
        if args.stage <= 3: logger.info("  -> Stage 3: chunk_stage2_output.py")
        if args.stage <= 4 and not args.skip_embeddings:
            logger.info("  -> Stage 4: embeddings_production.py (incremental)")
        return 0

    # Stages resolve their own data/<pipeline>/ paths from PIPELINE (via pipeline_config).
    if args.stage <= 1 and not run_stage(1, STAGE1_SCRIPT, env):
        return 1
    if args.stage <= 2 and not run_stage(2, STAGE2_SCRIPT, env):
        return 1
    if args.stage <= 3 and not run_stage(3, STAGE3_SCRIPT, env):
        return 1
    if args.stage <= 4 and not args.skip_embeddings and not run_stage(4, STAGE4_SCRIPT, env):
        return 1

    logger.info("")
    logger.info("=" * 70)
    logger.info("Pipeline completed successfully!")
    logger.info("=" * 70)
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("Output locations:")
    logger.info(f"  Stage 1 (Images):     {pc.paths.stage1}")
    logger.info(f"  Stage 2 (OCR):        {pc.paths.stage2}")
    logger.info(f"  Stage 3 (Chunks):     {pc.paths.chunks}")
    logger.info(f"  Stage 4 (Vector DB):  {pc.paths.db}")
    logger.info("")
    logger.info("To query the vector database:")
    logger.info(f"  PIPELINE={pc.PIPELINE} python 04_embeddings_and_kg/scripts/embeddings_production.py --query")
    return 0


if __name__ == "__main__":
    sys.exit(main())
