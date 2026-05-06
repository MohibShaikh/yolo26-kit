# Fixture asset attribution

| File | Source | License |
|---|---|---|
| `*/input.jpg` (coco_bus) | <https://ultralytics.com/images/bus.jpg> | Used as YOLO sample image; see Ultralytics for terms |
| `*/input.jpg` (coco_zidane) | <https://ultralytics.com/images/zidane.jpg> | As above |

Raw outputs and expected detections are derivative of the YOLO model used (see `meta.json` `model_name`), produced by Ultralytics under AGPL-3.0. yolo26-kit code does not redistribute model weights — only the runtime tensor outputs and decoded detections, used here for testing parity.
