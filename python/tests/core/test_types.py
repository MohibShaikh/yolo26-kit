from yolo26_kit.core.types import COCO_CLASSES, Detection


def test_coco_classes_count_and_zero_index():
    assert len(COCO_CLASSES) == 80
    assert COCO_CLASSES[0] == "person"
    assert COCO_CLASSES[79] == "toothbrush"


def test_detection_typeddict_round_trip():
    d: Detection = {"box": [1.0, 2.0, 3.0, 4.0], "score": 0.91, "class": 0, "label": "person"}
    assert d["class"] == 0
    assert d["label"] == "person"
