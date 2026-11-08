import ast
import json
import os
import stat
import subprocess
import sys

import pandas as pd
import pytest

from mdai import inference


def _write_config(version_path, content, name="config.yaml"):
    mdai_folder = version_path / "model" / ".mdai"
    mdai_folder.mkdir(parents=True, exist_ok=True)
    (mdai_folder / name).write_text(content)


@pytest.mark.parametrize(
    "base_image, expected",
    [
        ("py37", "3.7"),
        ("py38", "3.8"),
        ("py39", "3.9"),
        ("py310", "3.10"),
        ("py311", "3.11"),
        ("py312", "3.12"),
        ("py313", "3.13"),
        ("py314", "3.14"),
        ("PY310", "3.10"),
    ],
)
def test_get_python_version(tmp_path, base_image, expected):
    _write_config(tmp_path, f"base_image: {base_image}\n")
    assert inference.get_python_version(str(tmp_path)) == expected


@pytest.mark.parametrize("config", ["device_type: cpu\n", "base_image:\n", ""])
def test_get_python_version_defaults_to_newest(tmp_path, config):
    _write_config(tmp_path, config)
    assert inference.get_python_version(str(tmp_path)) == "3.14"


def test_get_python_version_reads_config_yml(tmp_path):
    _write_config(tmp_path, "base_image: py39\n", name="config.yml")
    assert inference.get_python_version(str(tmp_path)) == "3.9"


@pytest.mark.parametrize("base_image", ["nvidia", "custom", "py36", "py315"])
def test_get_python_version_unsupported(tmp_path, base_image):
    _write_config(tmp_path, f"base_image: {base_image}\n")
    with pytest.raises(Exception, match="not supported for local inference"):
        inference.get_python_version(str(tmp_path))


def test_env_exists_matches_exact_name(monkeypatch):
    envs = {"envs": ["/opt/conda", "/opt/conda/envs/mdai_abcd", "/home/u/.conda/envs/mdai_xyz"]}

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout=json.dumps(envs), stderr="")

    monkeypatch.setattr(inference.subprocess, "run", fake_run)
    assert inference.env_exists("mdai_xyz")
    assert inference.env_exists("mdai_abcd")
    assert not inference.env_exists("mdai_abc")


FAKE_CONDA = """\
#!{python}
# Minimal conda stand-in: records calls and runs `conda run` commands with the test interpreter.
# Envs are tracked in a JSON file as {{name: python_version}}. Set FAKE_CONDA_FAIL_PIP=1 to make
# pip installs fail.
import json, os, subprocess, sys

args = sys.argv[1:]
with open({log!r}, "a") as f:
    f.write(json.dumps(args) + "\\n")

envs_file = {envs!r}
envs = json.load(open(envs_file)) if os.path.exists(envs_file) else {{}}
if args[:3] == ["env", "list", "--json"]:
    print(json.dumps({{"envs": ["/fake/envs/" + e for e in envs]}}))
elif args[0] == "create":
    python = [a for a in args if a.startswith("python=")][0]
    envs[args[args.index("-n") + 1]] = python.split("=")[1]
    json.dump(envs, open(envs_file, "w"))
elif args[0] == "remove":
    envs.pop(args[args.index("-n") + 1], None)
    json.dump(envs, open(envs_file, "w"))
elif args[0] == "run":
    assert os.environ.get("PYTHONNOUSERSITE") == "1"
    name = args[args.index("-n") + 1]
    cmd = args[args.index("-n") + 2:]
    if cmd[:4] == ["python", "-m", "pip", "install"]:
        sys.exit(1 if os.environ.get("FAKE_CONDA_FAIL_PIP") else 0)
    if "sys.version_info" in cmd[-1]:
        print("output from an activate.d script")
        print("MDAI_PYTHON_VERSION=" + envs[name])
        sys.exit(0)
    assert cmd[0] == "python"
    sys.exit(subprocess.call([sys.executable] + cmd[1:]))
else:
    sys.exit(1)
"""

FAKE_MODEL = """\
class MDAIModel:
    def predict(self, data):
        return [
            {
                "type": "ANNOTATION",
                "study_uid": "1.2.3",
                "series_uid": "1.2.3.4",
                "instance_uid": "1.2.3.4.5",
                "class_index": 0,
                "probability": 0.9,
                "data": None,
            }
        ]
"""


@pytest.fixture
def fake_conda(tmp_path, monkeypatch):
    """Puts a fake `conda` on PATH. Returns a function that reads the recorded calls."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "conda_calls.log"
    log.write_text("")
    conda = bin_dir / "conda"
    conda.write_text(
        FAKE_CONDA.format(python=sys.executable, log=str(log), envs=str(tmp_path / "envs.json"))
    )
    conda.chmod(conda.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    def calls():
        lines = log.read_text().splitlines()
        log.write_text("")
        return [json.loads(line) for line in lines]

    return calls


def _make_model(tmp_path, version, base_image):
    model_path = tmp_path / "model"
    version_path = model_path / "source" / version
    _write_config(version_path, f"base_image: {base_image}\n")
    (version_path / "model" / ".mdai" / "mdai_deploy.py").write_text(FAKE_MODEL)
    (version_path / "model" / ".mdai" / "requirements.txt").write_text("")
    (model_path / "model.json").write_text(
        json.dumps(
            {
                "model": {
                    "id": "abc123",
                    "scope": "INSTANCE",
                    "labelClasses": [{"classIndex": 0, "labelId": "L_1"}],
                },
                "labels": [{"id": "L_1", "name": "Finding", "scope": "INSTANCE"}],
            }
        )
    )
    data_path = tmp_path / "dicom"
    data_path.mkdir(exist_ok=True)
    (data_path / "image.dcm").write_bytes(b"not parsed for INSTANCE scope")
    return str(model_path), str(data_path)


def test_run_inference_end_to_end(tmp_path, fake_conda):
    model_path, data_path = _make_model(tmp_path, "v1", "py310")

    inference.run_inference(model_path, data_path, model_version="v1")

    calls = fake_conda()
    assert ["create", "-n", "mdai_abc123", "python=3.10", "pip", "-y"] in calls
    assert any(c[:1] == ["run"] and "-r" in c for c in calls)

    df = pd.read_csv(os.path.join(model_path, "outputs_dicom.csv"))
    assert df["Label"].tolist() == ["Finding"]
    assert df["Probability"].tolist() == [0.9]

    # Second run reuses the env instead of creating it again.
    inference.run_inference(model_path, data_path, model_version="v1")
    calls = fake_conda()
    assert not any(c[0] == "create" for c in calls)
    assert not any("pip" in c for c in calls)


def test_run_inference_rejects_env_with_other_python(tmp_path, fake_conda):
    model_path, data_path = _make_model(tmp_path, "v1", "py37")
    inference.run_inference(model_path, data_path, model_version="v1")

    _make_model(tmp_path, "v2", "py310")
    with pytest.raises(Exception, match=r"uses Python 3\.7.*needs Python 3\.10.*delete_env"):
        inference.run_inference(model_path, data_path, model_version="v2")


def test_run_inference_removes_env_when_build_fails(tmp_path, fake_conda, monkeypatch):
    model_path, data_path = _make_model(tmp_path, "v1", "py39")
    monkeypatch.setenv("FAKE_CONDA_FAIL_PIP", "1")

    with pytest.raises(subprocess.CalledProcessError):
        inference.run_inference(model_path, data_path, model_version="v1")

    assert ["remove", "-n", "mdai_abc123", "--all", "-y"] in fake_conda()
    assert not inference.env_exists("mdai_abc123")


def test_env_exists_raises_when_conda_fails(monkeypatch):
    def fake_run(args, **kwargs):
        raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(inference.subprocess, "run", fake_run)
    with pytest.raises(subprocess.CalledProcessError):
        inference.env_exists("mdai_abc123")


def test_env_exists_raises_without_conda(monkeypatch):
    def fake_run(args, **kwargs):
        raise FileNotFoundError(args[0])

    monkeypatch.setattr(inference.subprocess, "run", fake_run)
    with pytest.raises(Exception, match="conda is required"):
        inference.env_exists("mdai_abc123")


def test_inference_module_runs_standalone_on_py37():
    # run_inference executes inference.py standalone in the model's env, which can be Python 3.7
    # and doesn't have mdai installed.
    with open(inference.__file__) as f:
        tree = ast.parse(f.read(), feature_version=(3, 7))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative import"
            names = [node.module]
        else:
            continue
        assert not any(n == "mdai" or n.startswith("mdai.") for n in names), names
