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
    [("py37", "3.7"), ("py38", "3.8"), ("py39", "3.9"), ("py310", "3.10"), ("PY310", "3.10")],
)
def test_get_python_version(tmp_path, base_image, expected):
    _write_config(tmp_path, f"base_image: {base_image}\n")
    assert inference.get_python_version(str(tmp_path)) == expected


def test_get_python_version_defaults_to_py37(tmp_path):
    _write_config(tmp_path, "device_type: cpu\n")
    assert inference.get_python_version(str(tmp_path)) == "3.7"


def test_get_python_version_reads_config_yml(tmp_path):
    _write_config(tmp_path, "base_image: py39\n", name="config.yml")
    assert inference.get_python_version(str(tmp_path)) == "3.9"


@pytest.mark.parametrize("base_image", ["nvidia", "custom", "py311"])
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
# Minimal conda stand-in: records calls, runs `conda run` commands with the test interpreter.
import json, os, subprocess, sys

args = sys.argv[1:]
with open({log!r}, "a") as f:
    f.write(json.dumps(args) + "\\n")

envs_file = {envs!r}
envs = json.load(open(envs_file)) if os.path.exists(envs_file) else []
if args[:3] == ["env", "list", "--json"]:
    print(json.dumps({{"envs": ["/fake/envs/" + e for e in envs]}}))
elif args[0] == "create":
    envs.append(args[args.index("-n") + 1])
    json.dump(envs, open(envs_file, "w"))
elif args[0] == "run":
    cmd = args[args.index("-n") + 2:]
    if cmd[:4] == ["python", "-m", "pip", "install"]:
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


def test_run_inference_end_to_end(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "conda_calls.log"
    conda = bin_dir / "conda"
    conda.write_text(
        FAKE_CONDA.format(python=sys.executable, log=str(log), envs=str(tmp_path / "envs.json"))
    )
    conda.chmod(conda.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    model_path = tmp_path / "model"
    version_path = model_path / "source" / "v1"
    _write_config(version_path, "base_image: py310\n")
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
    data_path.mkdir()
    (data_path / "image.dcm").write_bytes(b"not parsed for INSTANCE scope")

    inference.run_inference(str(model_path), str(data_path), model_version="v1")

    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert ["create", "-n", "mdai_abc123", "python=3.10", "pip", "-y"] in calls
    assert any(c[:1] == ["run"] and "-r" in c for c in calls)

    df = pd.read_csv(model_path / "outputs_dicom.csv")
    assert df["Label"].tolist() == ["Finding"]
    assert df["Probability"].tolist() == [0.9]

    # Second run reuses the env instead of creating it again.
    log.write_text("")
    inference.run_inference(str(model_path), str(data_path), model_version="v1")
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert not any(c[0] == "create" for c in calls)
    assert sum(c[0] == "run" for c in calls) == 1


def test_inference_module_has_no_mdai_imports():
    # run_inference executes inference.py standalone in the model's env, where mdai isn't installed.
    source = open(inference.__file__).read()
    assert "from ." not in source
    assert "import mdai\n" not in source and "from mdai " not in source
