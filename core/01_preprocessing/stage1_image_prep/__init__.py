"""Stage 1 — Image Preparation for Government Document OCR Pipeline."""

__all__ = ["ImagePrepPipeline"]


def __getattr__(name):
    # Rendering-only clients should not need OpenCV or the full prep pipeline.
    if name == "ImagePrepPipeline":
        from .pipeline import ImagePrepPipeline
        globals()[name] = ImagePrepPipeline
        return ImagePrepPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
