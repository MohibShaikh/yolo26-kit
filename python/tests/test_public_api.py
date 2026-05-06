def test_top_level_exports():
    import yolo26_kit
    expected = {
        "filter_e2e", "decode_detect", "e2e_to_v8_shape", "v8_shape_to_e2e",
        "letterbox_unmap", "normalize_output", "COCO_CLASSES",
        "__version__",
    }
    assert expected.issubset(set(dir(yolo26_kit)))
