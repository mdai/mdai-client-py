# Development

## Python version

[Pyenv](https://github.com/pyenv/pyenv) is recommended for managing python versions. Currently, python 3.7 is used for development.

## Dependencies

Uses [Pipenv](https://docs.pipenv.org). For initial setup, run:

```sh
# Install virtualenv in local directory
PIPENV_VENV_IN_PROJECT=1 pipenv install --dev
```

VSCode will automatically load the virtualenv. [flake8](http://flake8.pycqa.org) (linting) and [black](https://github.com/ambv/black) (formatter) are installed as dev dependencies in Pipenv. VSCode workspace settings are at `.vscode/`.
