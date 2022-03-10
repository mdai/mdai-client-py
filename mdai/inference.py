import os
import json
import sys
import shlex
import yaml
import zipfile
import pydicom
import pathlib
import subprocess
import pandas as pd
from tqdm import tqdm


def extract_zip(version_path, model_version):
    """Extracts the exported model zip file"""
    if not os.path.isdir(version_path):
        zip_file = version_path + ".zip"
        if os.path.isfile(zip_file):
            print(f"Extracting {model_version}.zip")
            with zipfile.ZipFile(zip_file, "r") as f:
                f.extractall(version_path)
        else:
            raise Exception(f"{model_version}.zip does not exist, please try a different version.")


def load_json(json_file):
    """Loads a json file"""
    with open(json_file) as f:
        json_file = json.load(f)
    return json_file


def parse_model_json(model_json):
    """Parses the model.json schema file. Returns model scope and output labels"""
    model = model_json["model"]
    model_scope = model["scope"]
    labels = {i["id"]: [i["name"], i["scope"]] for i in model_json["labels"]}
    labels = {i["classIndex"]: labels[i["labelId"]] for i in model["labelClasses"]}
    return model_scope, labels


def get_file_paths(root):
    """Yields all file paths recursively from root path, filtering on DICOM extension."""
    if os.path.isfile(root):
        yield root
    else:
        for item in os.scandir(root):
            if item.is_file():
                if os.path.splitext(item.path)[1] == ".dcm":
                    yield item.path
            elif item.is_dir():
                yield from get_file_paths(item.path)


def process_file(path):
    """Returns each instance in raw bytes format"""
    instance = {}
    with open(path, "rb") as f:
        instance["content"] = f.read()
        instance["content_type"] = "application/dicom"
    return instance


def get_scope_files(file_paths, scope):
    """Returns aggregated list of files based on input scope of the model"""
    scope_map = {"STUDY": "StudyInstanceUID", "SERIES": "SeriesInstanceUID"}
    vals = {}
    for path in file_paths:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        uid = ds.get(scope_map[scope])
        if uid not in vals:
            vals[uid] = {"files": [], "annotations": [], "args": {}}
        vals[uid]["files"].append(process_file(path))
    del ds
    return list(vals.values())


def process_data(path, model_scope):
    """Returns processed data in the correct input format for models"""
    file_paths = list(get_file_paths(path))
    if model_scope in ("STUDY", "SERIES"):
        return get_scope_files(file_paths, model_scope)
    else:
        data = []
        for path in file_paths:
            val = {"files": [], "annotations": [], "args": {}}
            val["files"].append(process_file(path))
            data.append(val)
        return data


def run_model(data_path, model_path, model_scope):
    """Prepares inputs and run the MDAI model"""
    print("Preparing inputs", flush=True)
    input_data = process_data(data_path, model_scope)

    sys.path.insert(0, os.path.join(model_path, ".mdai"))
    from mdai_deploy import MDAIModel

    model = MDAIModel()

    outputs = []
    for data in tqdm(input_data, desc="Running inference"):
        outputs.append(model.predict(data))
    outputs = [val for output in outputs for val in output]
    return outputs


def env_exists(env_name):
    """Checks if conda env alreay exists to prevent duplicate builds"""
    command = shlex.split(f"/bin/bash -c 'conda env list | grep {env_name}'")
    try:
        subprocess.run(command, capture_output=True, text=True, check=True)
    except Exception:
        return False
    return True


def is_py37(version_path):
    """Checks if base_image is py37 in config.yaml"""
    config_path = os.path.join(version_path, "model", ".mdai", "config.yaml")
    with open(config_path, "r") as f:
        config_file = yaml.safe_load(f)

    if config_file["base_image"] == "py37":
        return True
    return False


def infer(model_path, data_path, model_version):
    """Helper function for processing inputs and running the model"""
    file_name = os.path.splitext(data_path)[0].split("/")[-1]
    version_path = os.path.join(model_path, "source", model_version)

    model_json = load_json(os.path.join(model_path, "model.json"))
    model_scope, labels = parse_model_json(model_json)
    model_inference_path = os.path.join(version_path, "model")

    outputs = run_model(data_path, model_inference_path, model_scope)
    columns = [
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "SOPInstanceUID",
        "Label",
        "Probability",
        "Data",
        "Scope",
    ]
    df = pd.DataFrame(columns=columns)

    for output in outputs:
        if output.get("type") == "ANNOTATION":
            label_details = labels[output.get("class_index")]
            row = {
                "StudyInstanceUID": output.get("study_uid"),
                "SeriesInstanceUID": output.get("series_uid"),
                "SOPInstanceUID": output.get("instance_uid"),
                "Label": label_details[0],
                "Probability": output.get("probability"),
                "Scope": label_details[1],
                "Data": [output.get("data")],
            }
            df = pd.concat([df, pd.DataFrame(row, index=[0])], ignore_index=True, axis=0)
    df.to_csv(os.path.join(model_path, f"outputs_{file_name}.csv"), index=False)
    print("Done!", flush=True)


def delete_env(model_path):
    """
    Delete the conda env created by previous model runs.

    Args:
        model_path: Path to the exported MDAI `model` folder
    """
    model_json = load_json(os.path.join(model_path, "model.json"))
    model_id = model_json["model"]["id"]
    env_name = f"mdai_{model_id}"

    print(f"Deleting conda env {env_name}")
    subprocess.run(shlex.split(f'/bin/bash -c "conda env remove -n {env_name}"'))


def run_inference(model_path, data_path, model_version="v1"):
    """
    Run exported MDAI models locally. Returns a csv of model outputs.

    Args:
        model_path: Path to the exported and extracted MDAI `model` folder
        data_path: Path to the input DICOM files
        model_version: Version of the downloaded model to run. Default 'v1'

    """
    model_path = pathlib.Path(model_path)
    data_path = pathlib.Path(data_path)
    version_path = os.path.join(model_path, "source", model_version)

    model_json = load_json(os.path.join(model_path, "model.json"))
    model_id = model_json["model"]["id"]
    env_name = f"mdai_{model_id}"

    if not os.path.exists(model_path):
        raise Exception(" Path for extracted model does not exist.")

    if not os.path.exists(data_path):
        raise Exception("Path for input data does not exist.")

    extract_zip(version_path, model_version)

    if not is_py37(version_path):
        raise Exception(
            "Custom Dockerfiles and NVIDIA base images are not currently supported for local inference."
        )

    if env_exists(env_name):
        print(f"Loading conda env {env_name}")
        command = shlex.split(
            r'''/bin/bash -c "source activate {} && \
                python -c 'import mdai; mdai.infer(\"{}\", \"{}\", \"{}\")'"'''.format(
                env_name, model_path, data_path, model_version,
            )
        )
    else:
        command = shlex.split(
            r'''/bin/bash -c "conda create -n {} python=3.7 pip -y  && \
                source activate {} && \
                pip install numpy tqdm pandas mdai ipykernel pyyaml pydicom==2.1.2 h5py==2.10.0 && \
                pip install -r {} && \
                python -c 'import mdai; mdai.infer(\"{}\", \"{}\", \"{}\")'"'''.format(
                env_name,
                env_name,
                os.path.join(version_path, "model", ".mdai", "requirements.txt"),
                model_path,
                data_path,
                model_version,
            )
        )
    subprocess.run(command)
