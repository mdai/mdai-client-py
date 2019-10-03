import os
import zipfile
import shutil
import requests
import pytest

from mdai.preprocess import Project

FIXTURES_BASE_URL = "https://storage.googleapis.com/mdai-app-data/test-fixtures/mdai-client-py/"
IMG_FILE = "mdai_staging_project_bwRnkNW2_images_2018-08-25-192424.zip"
ANNO_FILE = "mdai_staging_project_bwRnkNW2_annotations_labelgroup_all_2018-08-25-204133.json"
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def download_file(url):
    local_filename = os.path.join(TEST_DATA_DIR, url.split("/")[-1])
    r = requests.get(url, stream=True)
    if r.status_code == requests.codes.ok:
        with open(local_filename, "wb") as f:
            shutil.copyfileobj(r.raw, f)
        return local_filename
    else:
        r.raise_for_status()


@pytest.fixture
def p():

    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    annotations_fp = download_file(FIXTURES_BASE_URL + ANNO_FILE)
    images_dir_zipped = download_file(FIXTURES_BASE_URL + IMG_FILE)
    with zipfile.ZipFile(images_dir_zipped) as zf:
        zf.extractall(TEST_DATA_DIR)
    (images_dir, ext) = os.path.splitext(images_dir_zipped)

    p = Project(annotations_fp=annotations_fp, images_dir=images_dir)
    return p
