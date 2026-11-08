# NOTE: run_inference() executes this file with the model's own Python interpreter (as old as 3.7,
# see PYTHON_VERSIONS), so it must stay compatible with Python 3.7 and must not import other mdai
# modules.
import os
import json
import sys
import shlex
import yaml
import zipfile
import pydicom
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
    """Checks if conda env already exists to prevent duplicate builds"""
    # Raise rather than return False on errors: a False answer leads to (re)creating the env, and
    # a failed build removes it.
    try:
        result = subprocess.run(
            ["conda", "env", "list", "--json"], capture_output=True, text=True, check=True
        )
    except FileNotFoundError:
        raise Exception("conda is required for local inference but was not found on PATH.")
    envs = json.loads(result.stdout).get("envs", [])
    return any(os.path.basename(os.path.normpath(env)) == env_name for env in envs)


# Python version for each supported `base_image` in the model's .mdai/config.yaml, oldest first.
# Mirrors the python base images used when building models on MD.ai, plus py37 and py38, which
# MD.ai no longer builds but older exported models may still use.
PYTHON_VERSIONS = {
    "py37": "3.7",
    "py38": "3.8",
    "py39": "3.9",
    "py310": "3.10",
    "py311": "3.11",
    "py312": "3.12",
    "py313": "3.13",
    "py314": "3.14",
}
# Used when config.yaml does not set base_image, as when building models on MD.ai: the newest
# version.
DEFAULT_BASE_IMAGE = list(PYTHON_VERSIONS)[-1]

# Packages installed into the conda env before the model's own requirements.txt, so the model's
# pins win. They cover what this module needs, plus the DICOM decoders that MD.ai model images
# include.
BASE_REQUIREMENTS = [
    "numpy",
    "pandas",
    "tqdm",
    "pyyaml",
    "pydicom>=2.4,<4",
    "pylibjpeg>=1.4,<3",
    "pylibjpeg-libjpeg>=1.3,<3",
    "pylibjpeg-openjpeg>=1.3,<3",
    "pylibjpeg-rle>=1.3,<3",
]


def get_python_version(version_path):
    """Returns the Python version for the model's `base_image` in .mdai/config.yaml"""
    mdai_folder = os.path.join(version_path, "model", ".mdai")
    config = {}
    for name in ("config.yaml", "config.yml"):
        config_path = os.path.join(mdai_folder, name)
        if os.path.isfile(config_path):
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}
            break

    base_image = str(config.get("base_image") or DEFAULT_BASE_IMAGE).lower()
    if base_image not in PYTHON_VERSIONS:
        raise Exception(
            f"base_image '{base_image}' is not supported for local inference. Supported base "
            f"images: {', '.join(PYTHON_VERSIONS)}. Custom Dockerfiles and NVIDIA base images are "
            "not supported."
        )
    return PYTHON_VERSIONS[base_image]


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

    Requires conda 4.9 or later. The model runs in a conda env named `mdai_<model id>` with the
    Python version of the model's `base_image` (py37 through py314). The env is created on
    the first run and reused afterwards; remove it with `delete_env`.

    Args:
        model_path: Path to the exported and extracted MDAI `model` folder
        data_path: Path to the input DICOM files
        model_version: Version of the downloaded model to run. Default 'v1'

    """
    model_path = os.path.abspath(model_path)
    data_path = os.path.abspath(data_path)
    version_path = os.path.join(model_path, "source", model_version)

    if not os.path.exists(model_path):
        raise Exception("Path for extracted model does not exist.")

    if not os.path.exists(data_path):
        raise Exception("Path for input data does not exist.")

    model_json = load_json(os.path.join(model_path, "model.json"))
    model_id = model_json["model"]["id"]
    env_name = f"mdai_{model_id}"

    extract_zip(version_path, model_version)
    python_version = get_python_version(version_path)

    if env_exists(env_name):
        print(f"Loading conda env {env_name}")
        env_python_version = _env_python_version(env_name)
        if env_python_version != python_version:
            raise Exception(
                f"Conda env {env_name} uses Python {env_python_version}, but model version "
                f"{model_version} needs Python {python_version}. Remove the env with "
                "mdai.delete_env(model_path) and run again."
            )
    else:
        print(f"Creating conda env {env_name} with Python {python_version}")
        requirements_path = os.path.join(version_path, "model", ".mdai", "requirements.txt")
        try:
            subprocess.run(
                ["conda", "create", "-n", env_name, f"python={python_version}", "pip", "-y"],
                check=True,
            )
            _conda_run(env_name, ["python", "-m", "pip", "install", *BASE_REQUIREMENTS])
            if os.path.isfile(requirements_path):
                _conda_run(env_name, ["python", "-m", "pip", "install", "-r", requirements_path])
        except BaseException:
            # Don't leave a half-built env behind (also on Ctrl-C): it would be reused by the
            # next run.
            subprocess.run(["conda", "remove", "-n", env_name, "--all", "-y"])
            raise

    # Run this file with the env's interpreter instead of importing mdai there: the env's Python
    # can be older than what the installed mdai release supports. run_path doesn't add this
    # file's directory to sys.path, so the model can't accidentally import mdai's modules.
    script = (
        "import runpy, sys; "
        "sys.argv = sys.argv[1:]; "
        "runpy.run_path(sys.argv[0], run_name='__main__')"
    )
    _conda_run(
        env_name,
        ["python", "-c", script, os.path.abspath(__file__), model_path, data_path, model_version],
    )


def _conda_run(env_name, args, **kwargs):
    # Keep the user's site-packages (~/.local) for the same Python version out of the env.
    env = {**os.environ, "PYTHONNOUSERSITE": "1"}
    return subprocess.run(
        ["conda", "run", "--no-capture-output", "-n", env_name, *args],
        check=True,
        env=env,
        **kwargs,
    )


def _env_python_version(env_name):
    """Returns the major.minor Python version of a conda env"""
    result = _conda_run(
        env_name,
        ["python", "-c", "import sys; print('MDAI_PYTHON_VERSION=%d.%d' % sys.version_info[:2])"],
        capture_output=True,
        text=True,
    )
    # Activation scripts in the env may print to stdout too.
    for line in result.stdout.splitlines():
        if line.startswith("MDAI_PYTHON_VERSION="):
            return line.split("=", 1)[1].strip()
    raise Exception(f"Could not determine the Python version of conda env {env_name}.")


if __name__ == "__main__":
    infer(*sys.argv[1:4])
