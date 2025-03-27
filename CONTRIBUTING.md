# Development

## Python version

[Pyenv](https://github.com/pyenv/pyenv) is recommended for managing python versions. Currently, python 3.8+ is used for development.

## Virtualenv

Uses [Poetry](https://poetry.eustace.io/docs/). For initial setup, run:

```sh
# Install poetry (1.0+)
pip install -U poetry

# Configure poetry to install virtualenv in local directory
poetry config virtualenvs.create true
poetry config virtualenvs.in-project true

# Install virtualenv in local directory
poetry install
```

VSCode will automatically load the virtualenv. [ruff](https://github.com/charliermarsh/ruff) (linting) and [black](https://github.com/ambv/black) (formatter) are installed as dev dependencies.

To activate the local virtualenv:

```sh
source .venv/bin/activate
# or
poetry shell
```

## Creating a new release

Update the library version in the following files with every PR according to [semver](https://semver.org/) guidelines -

- [pyproject.toml](https://github.com/mdai/mdai-client-py/blob/master/pyproject.toml#L17)

Add a new tag -

```sh
git tag -a <tagname> -m "tag message"
```

Push the new tag -

```sh
git push origin <tagname>
```
