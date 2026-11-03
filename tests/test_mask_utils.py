from types import SimpleNamespace

import numpy as np

from mdai import visualize
from mdai.utils.common_utils import convert_mask_annotation_to_array, convert_mask_data

# Non-square image so that transposed height/width would be detected.
HEIGHT, WIDTH = 40, 60


def _square_with_hole():
    """Mask data: a filled square (foreground) with a smaller square hole (background)."""
    expected = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    expected[5:30, 10:50] = 1
    expected[12:20, 20:35] = 0
    return convert_mask_data(expected), expected


def test_convert_mask_annotation_to_array_non_square():
    data, expected = _square_with_hole()
    assert data["foreground"] and data["background"]

    row = SimpleNamespace(width=WIDTH, height=HEIGHT, data=data)
    mask = convert_mask_annotation_to_array(row)

    assert mask.shape == (HEIGHT, WIDTH)
    # Contour-based roundtrip can differ at the boundary of the hole; check interior pixels.
    assert mask[7, 12] == 1
    assert mask[28, 48] == 1
    assert mask[15, 27] == 0
    assert mask[35, 55] == 0


def test_convert_mask_annotation_to_array_missing_background():
    data = {"foreground": [[[10, 5], [49, 5], [49, 29], [10, 29]]], "background": None}
    row = SimpleNamespace(width=WIDTH, height=HEIGHT, data=data)
    mask = convert_mask_annotation_to_array(row)

    assert mask.shape == (HEIGHT, WIDTH)
    assert mask[15, 27] == 1
    assert mask[35, 55] == 0


class _FakeDataset:
    def __init__(self, annotations, modes):
        self._annotations = annotations
        self._modes = modes

    def get_annotations_by_image_id(self, image_id):
        return self._annotations

    def label_id_to_class_annotation_mode(self, label_id):
        return self._modes[label_id]

    def label_id_to_class_id(self, label_id):
        return {"L_bbox": 1, "L_mask": 2}[label_id]


def test_load_mask_with_mask_annotation(monkeypatch):
    data, _ = _square_with_hole()
    annotations = [
        {"labelId": "L_bbox", "data": {"x": 1, "y": 1, "width": 3, "height": 3}},
        {"labelId": "L_mask", "data": data},
    ]
    dataset = _FakeDataset(annotations, {"L_bbox": "bbox", "L_mask": "mask"})
    monkeypatch.setattr(
        visualize, "load_dicom_image", lambda image_id: np.zeros((HEIGHT, WIDTH), dtype=np.int16)
    )

    mask, class_ids = visualize.load_mask("image.dcm", dataset)

    assert mask.shape == (HEIGHT, WIDTH, 2)
    assert class_ids.tolist() == [1, 2]
    # bbox instance stays on its own channel
    assert mask[2, 2, 0] and not mask[2, 2, 1]
    # mask instance: foreground filled, hole cleared, outside empty
    assert mask[7, 12, 1]
    assert not mask[15, 27, 1]
    assert not mask[35, 55, 1]
    assert not mask[7, 12, 0]
