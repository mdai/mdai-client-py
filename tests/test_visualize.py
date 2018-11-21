import pytest
from mdai import visualize
import numpy as np

# TODO: test load_dicom_image (with RGB option or not)


def test_visualize(p):
    labels_dict = {
        "L_egJRyg": 1,  # bounding box
        "L_MgevP2": 2,  # polygon
        "L_D21YL2": 3,  # freeform
        "L_lg7klg": 4,  # line
        "L_eg69RZ": 5,  # location
        "L_GQoaJg": 6,  # global_image
        "L_JQVWjZ": 7,  # global_series
        "L_3QEOpg": 8,  # global_exam
    }

    p.set_labels_dict(labels_dict)

    ct_dataset = p.get_dataset_by_id("D_qGQdpN")
    ct_dataset.prepare()

    # image with multiple annotations
    image_id = ct_dataset.get_image_ids()[7]

    grey_image = visualize.load_dicom_image(image_id)
    rgb_image = visualize.load_dicom_image(image_id, to_RGB=True)
    scaled_image_1 = visualize.load_dicom_image(image_id, rescale=True)

    assert np.amax(grey_image) == 701
    assert np.amax(scaled_image_1) == 255

    assert grey_image.shape == (256, 256)
    assert rgb_image.shape == (256, 256, 3)

    scaled_image_2, gt_class_id, gt_bbox, gt_mask = visualize.get_image_ground_truth(
        image_id, ct_dataset
    )

    assert len(gt_class_id) == len(gt_class_id)
    assert gt_mask.shape == (256, 256, 5)
