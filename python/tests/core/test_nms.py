import numpy as np

from yolo26_kit.core.nms import class_aware_nms


def test_nms_drops_overlapping_same_class() -> None:
    boxes = np.asarray([
        [10, 10, 50, 50],
        [12, 12, 52, 52],
        [100, 100, 150, 150],
    ], dtype=np.float32)
    scores = np.asarray([0.9, 0.8, 0.7], dtype=np.float32)
    classes = np.asarray([0, 0, 0], dtype=np.int32)
    keep = class_aware_nms(boxes, scores, classes, iou_threshold=0.45)
    assert keep.tolist() == [0, 2]


def test_nms_keeps_overlapping_different_classes() -> None:
    boxes = np.asarray([
        [10, 10, 50, 50],
        [12, 12, 52, 52],
    ], dtype=np.float32)
    scores = np.asarray([0.9, 0.8], dtype=np.float32)
    classes = np.asarray([0, 1], dtype=np.int32)
    keep = class_aware_nms(boxes, scores, classes, iou_threshold=0.45)
    assert sorted(keep.tolist()) == [0, 1]


def test_nms_empty_input() -> None:
    keep = class_aware_nms(
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
        np.zeros((0,), dtype=np.int32),
    )
    assert keep.shape == (0,)
