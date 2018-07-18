# mdai-client-py

MD.ai Python client library

**Currently pre-alpha -- API may change significantly in future releases.**

## Development

### Python version

[Pyenv](https://github.com/pyenv/pyenv) is recommended for managing python versions. Currently, python 3.6 is used for development.

### Dependencies

[Pipenv](https://docs.pipenv.org)

For initial setup, run:

```sh
# Install virtualenv in local directory
export PIPENV_VENV_IN_PROJECT=1

pipenv install --dev --three
```

VSCode should automatically load the virtualenv located in the current directory. If not, go to `Python: Select Interpreter` from the command palette.

### Formatting

[black](https://github.com/ambv/black)

Installed as dev dependencies in Pipenv. To use, add the following to user settings:

```json
{
  // ...
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length=100"]
  // ...
}
```

### Linting

[Pycodestyle](http://pycodestyle.pycqa.org/en/latest/)

Installed as dev dependencies in Pipenv. To use, add the following to user settings:

```json
{
  // ...
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.flake8Enabled": false,
  "python.linting.pep8Enabled": true,
  "python.linting.pep8Path": "pycodestyle",
  "python.linting.pep8Args": ["--max-line-length=100"]
  // ...
}
```

---

&copy; 2018 MD.ai, Inc.
