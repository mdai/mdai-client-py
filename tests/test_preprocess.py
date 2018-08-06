import pytest


def test_project(hello_world_project):

    # label groups
    label_groups = hello_world_project.get_label_groups()
    assert label_groups[0].id == "G_3lv"

    datasets = hello_world_project.get_datasets()
    assert len(datasets) == 3

    # assert project.selected_label_ids == None

    # label_ids = ["L_yxv", "L_dyy"]
    # project.set_label_ids(label_ids)
    # assert project.selected_label_ids == label_ids


def test_dataset(hello_world_project):
    labels_dict = {"L_yxv": 0, "L_dyy": 1}  # chest, abdomen
    hello_world_project.set_labels_dict(labels_dict)
    train_dataset = hello_world_project.get_dataset_by_name("TRAIN")
    train_dataset.prepare()
    train_image_ids = train_dataset.get_image_ids()
    assert len(train_image_ids) == 65


# TODO: add more tests
# - check class id to class text
# - check filtered vs. unfiltered
# - check generate image ids (case where SOP UI is not available?)
