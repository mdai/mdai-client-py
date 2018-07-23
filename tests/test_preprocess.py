import pytest
import os
import urllib
import zipfile
import shutil
import requests

import pydicom

from mdai.preprocess import Project


HELLOWORLD_IMAGES_URL = "https://s3.amazonaws.com/mdai-test-data-public/mdai_public_project_PVq9raBJ_dataset_all_2018-07-17-101532.zip"
HELLOWORLD_ANNO_URL = "https://s3.amazonaws.com/mdai-test-data-public/mdai_public_project_PVq9raBJ_dataset_all_labelgroup_all_2018-07-17-101553.json"


def download_file(url):
    local_filename = os.path.join("tests/data", url.split("/")[-1])
    r = requests.get(url, stream=True)
    with open(local_filename, "wb") as f:
        shutil.copyfileobj(r.raw, f)
    return local_filename


@pytest.fixture
def project():
    annotations_fp = download_file(HELLOWORLD_ANNO_URL)
    images_dir_zipped = download_file(HELLOWORLD_IMAGES_URL)

    print(images_dir_zipped)
    with zipfile.ZipFile(images_dir_zipped) as zf:
        zf.extractall()
    (images_dir, ext) = os.path.splitext(images_dir_zipped)

    p = Project(annotations_fp=annotations_fp, images_dir=images_dir)
    return p


def test_project(project):

    # label groups
    label_groups = project.get_label_groups()
    assert label_groups[0].id == "G_3lv"

    datasets = project.get_datasets()
    assert len(datasets) == 3

    assert project.selected_label_ids == None

    label_ids = ["L_yxv", "L_dyy"]
    project.set_label_ids(label_ids)
    assert project.selected_label_ids == label_ids


def test_dataset(project):
    label_ids = ["L_yxv", "L_dyy"]
    project.set_label_ids(label_ids)
    train_dataset = project.get_dataset_by_name("TRAIN")
    train_dataset.prepare()
    train_image_ids = train_dataset.get_image_ids()
    assert len(train_image_ids) == 65

