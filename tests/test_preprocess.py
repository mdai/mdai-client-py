import pytest
from mdai.utils.common_utils import train_test_split


def test_project(p):

    # label groups
    label_groups = p.get_label_groups()

    assert label_groups[0].id == "G_L3dP31"
    assert label_groups[1].id == "G_WVRrVJ"


def test_dataset(p):

    # two datasets
    datasets = p.get_datasets()
    assert len(datasets) == 2

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

    assert p.get_label_id_annotation_mode("L_MgevP2") == "polygon"
    assert p.get_label_id_annotation_mode("L_3QEOpg") is None

    ct_dataset = p.get_dataset_by_id("D_qGQdpN")
    ct_dataset.prepare()

    xray_dataset = p.get_dataset_by_id("D_0Z4nDG")
    xray_dataset.prepare()

    assert ct_dataset.classes_dict == xray_dataset.classes_dict

    image_ids = ct_dataset.get_image_ids()
    assert len(image_ids) == len(ct_dataset.imgs_anns_dict.keys())

    image_id = ct_dataset.get_image_ids()[7]

    ann_mode = [
        (ct_dataset.label_id_to_class_annotation_mode(ann["labelId"]), ann["labelId"])
        for ann in ct_dataset.imgs_anns_dict[image_id]
    ]

    assert ann_mode == [
        ("line", "L_lg7klg"),
        ("polygon", "L_MgevP2"),
        ("freeform", "L_D21YL2"),
        ("bbox", "L_egJRyg"),
        ("location", "L_eg69RZ"),
        (None, "L_3QEOpg"),
    ]

    train_ds, valid_ds = train_test_split(ct_dataset, shuffle=False, validation_split=0.2)

    assert len(train_ds.get_image_ids()) == 9
    assert len(valid_ds.get_image_ids()) == 3
