import os
import urllib
import zipfile
import shutil
import requests
import pytest

from mdai.preprocess import Project

HELLOWORLD_IMAGES_URL = "https://s3.amazonaws.com/mdai-test-data-public/mdai_public_project_PVq9raBJ_dataset_all_2018-07-17-101532.zip"
HELLOWORLD_ANNO_URL = "https://s3.amazonaws.com/mdai-test-data-public/mdai_public_project_PVq9raBJ_dataset_all_labelgroup_all_2018-07-17-101553.json"
TESTS_DATA_FP = "tests/data"


def download_file(url):
    local_filename = os.path.join(TESTS_DATA_FP, url.split("/")[-1])
    r = requests.get(url, stream=True)
    if r.status_code == requests.codes.ok:
        with open(local_filename, "wb") as f:
            shutil.copyfileobj(r.raw, f)
        return local_filename
    else:
        r.raise_for_status()


@pytest.fixture
def hello_world_project():

    os.makedirs(TESTS_DATA_FP, exist_ok=True)
    annotations_fp = download_file(HELLOWORLD_ANNO_URL)
    images_dir_zipped = download_file(HELLOWORLD_IMAGES_URL)
    with zipfile.ZipFile(images_dir_zipped) as zf:
        zf.extractall(TESTS_DATA_FP)
    (images_dir, ext) = os.path.splitext(images_dir_zipped)

    p = Project(annotations_fp=annotations_fp, images_dir=images_dir)
    return p
