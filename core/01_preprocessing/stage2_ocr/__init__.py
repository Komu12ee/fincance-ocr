"""Stage 2 — OCR and Structure Extraction with Docling."""

__all__ = ["OCRPipeline"]


def __getattr__(name):
    # Remote GPU OCR can reuse the data models without importing Docling.
    if name == "OCRPipeline":
        from .pipeline import OCRPipeline
        globals()[name] = OCRPipeline
        return OCRPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
