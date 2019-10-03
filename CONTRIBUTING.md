# Development

## Python version

[Pyenv](https://github.com/pyenv/pyenv) is recommended for managing python versions. Currently, python 3.7 is used for development.

## Virtualenv

Uses [Poetry](https://poetry.eustace.io/docs/). For initial setup, run:

```sh
# Configure poetry to install virtualenv in local directory
poetry config settings.virtualenvs.create true
poetry config settings.virtualenvs.in-project true

# Install virtualenv in local directory
poetry install
```

VSCode will automatically load the virtualenv. [flake8](http://flake8.pycqa.org) (linting) and [black](https://github.com/ambv/black) (formatter) are installed as dev dependencies.

Recommended VS Code workspace settings (`.vscode/settings.json`):

```json
{
  "python.pythonPath": ".venv/bin/python3.7",
  "python.linting.pylintEnabled": false,
  "python.linting.flake8Enabled": true,
  "python.linting.flake8Args": ["--max-line-length=100"],
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length=100"],
  "editor.formatOnSave": true
}
```

To activate the local virtualenv:

```sh
source .venv/bin/activate
```
