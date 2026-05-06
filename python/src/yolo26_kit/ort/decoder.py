"""Auto-routing wrapper around onnxruntime.InferenceSession."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from ..core.decode_raw import decode_detect
from ..core.filter_e2e import filter_e2e
from ..core.letterbox import letterbox_unmap
from ..core.normalize import normalize_output
from ..core.types import COCO_CLASSES
from .image_io import to_hwc_uint8
from .preprocess import letterbox_forward


def from_ort(session_or_path: Any, *, providers: list[str] | None = None) -> Decoder:
    if isinstance(session_or_path, (str, Path)):
        try:
            import onnxruntime as ort  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError("install yolo26-kit[ort] to use Decoder") from e
        session = ort.InferenceSession(
            str(session_or_path),
            providers=providers or ort.get_available_providers(),
        )
    else:
        session = session_or_path
    return Decoder(session)


class Decoder:
    def __init__(self, session: Any) -> None:
        self._session = session
        outs = session.get_outputs()
        if len(outs) != 1:
            raise ValueError(
                f"expected single output tensor; got {len(outs)}. "
                "Is this a YOLO26 detect ONNX?",
            )
        self._output_name = outs[0].name
        self._input_name = session.get_inputs()[0].name
        shape = tuple(outs[0].shape)
        self.is_e2e: bool = bool(shape and shape[-1] == 6)
        provs = session.get_providers() if hasattr(session, "get_providers") else []
        self.backend: str = provs[0] if provs else "unknown"

    def predict(
        self,
        image: Any,
        conf: float = 0.25,
        classes: Iterable[int] | None = None,
        format: Literal["dict", "arrays"] = "dict",
    ) -> Any:
        hwc = to_hwc_uint8(image)
        nchw, meta = letterbox_forward(hwc, target=640)
        out = self._session.run([self._output_name], {self._input_name: nchw})[0]
        out = normalize_output(out)

        if self.is_e2e:
            dets = filter_e2e(out, conf=conf, classes=classes, format="arrays")
        else:
            dets = decode_detect(out, conf=conf, classes=classes, format="arrays")

        assert isinstance(dets, dict)
        boxes = dets["boxes"]
        if boxes.size:
            boxes = letterbox_unmap(
                boxes,
                orig_size=meta["orig_size"],
                lb_size=meta["lb_size"],
                scale=meta["scale"],
                pad=meta["pad"],
            )
        if format == "arrays":
            return {"boxes": boxes, "scores": dets["scores"], "classes": dets["classes"]}

        return [
            {
                "box": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                "score": float(s),
                "class": int(c),
                "label": COCO_CLASSES[int(c)],
            }
            for b, s, c in zip(boxes, dets["scores"], dets["classes"], strict=True)
        ]
