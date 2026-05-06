"""yolo26-kit: YOLO26 ↔ YOLOv8-pipeline bridge."""
from typing import Any

from .core.decode_raw import decode_detect
from .core.filter_e2e import filter_e2e
from .core.letterbox import letterbox_unmap
from .core.normalize import normalize_output
from .core.shapes import e2e_to_v8_shape, v8_shape_to_e2e
from .core.types import COCO_CLASSES, Detection

__version__ = "0.1.0"

__all__ = [
    "COCO_CLASSES",
    "Decoder",
    "Detection",
    "__version__",
    "decode_detect",
    "e2e_to_v8_shape",
    "filter_e2e",
    "from_ort",
    "letterbox_unmap",
    "normalize_output",
    "v8_shape_to_e2e",
]


def __getattr__(name: str) -> Any:  # PEP 562 lazy export
    if name in {"from_ort", "Decoder"}:
        from .ort.decoder import Decoder, from_ort
        return {"from_ort": from_ort, "Decoder": Decoder}[name]
    raise AttributeError(f"module 'yolo26_kit' has no attribute {name!r}")
