# Development

## Python version

[Pyenv](https://github.com/pyenv/pyenv) is recommended for managing python versions. Currently, python 3.6 is used for development.

## Dependencies

Uses [Pipenv](https://docs.pipenv.org). For initial setup, run:

```sh
# Install virtualenv in local directory
PIPENV_VENV_IN_PROJECT=1 pipenv install --dev
```

VSCode will automatically load the virtualenv. [Pycodestyle](http://pycodestyle.pycqa.org/en/latest/) (linting) and [black](https://github.com/ambv/black) (formatter) are installed as dev dependencies in Pipenv. To use, VSCode workspace settings should look like the following:

```json
{
  "python.formatting.blackArgs": ["--line-length=100"],
  "python.formatting.provider": "black",
  "python.linting.pep8Args": ["--max-line-length=100"],
  "python.linting.pep8Enabled": true,
  "python.linting.pep8Path": "pycodestyle",
  "python.linting.pylintEnabled": false,
  "python.pythonPath": ".venv/bin/python"
}
```
