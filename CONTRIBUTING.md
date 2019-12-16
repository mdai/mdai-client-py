# Development

## Python version

[Pyenv](https://github.com/pyenv/pyenv) is recommended for managing python versions. Currently, python 3.7 is used for development.

## Virtualenv

Uses [Poetry](https://poetry.eustace.io/docs/). For initial setup, run:

```sh
# Install poetry
pip install poetry==1.0.0

# Configure poetry to install virtualenv in local directory
poetry config virtualenvs.create true
poetry config virtualenvs.in-project true

# Install virtualenv in local directory
poetry install
```

VSCode will automatically load the virtualenv. [flake8](http://flake8.pycqa.org) (linting) and [black](https://github.com/ambv/black) (formatter) are installed as dev dependencies.

To activate the local virtualenv:

```sh
source .venv/bin/activate
```
