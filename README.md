# mdai-client-py

MD.ai Python client library

## Documentation

Coming soon...

## Development

-**Python version management** 
-	
-Pyenv is recommended for managing python versions. 	
-See tutorial and installation guide [here](https://amaral.northwestern.edu/resources/guides/pyenv-tutorial). 	
-	
-Current python version to install is 3.6.5. 	
-```sh	
-  pyenv install 3.6.5	
-```	
-However, there are several system dependecies that need to be installed first [link](https://github.com/pyenv/pyenv/wiki/common-build-problems). Install these first: 	
-	
-```sh 	
-  sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev 	
-  xz-utils tk-dev libffi-dev	
-``` 	
**Dependencies**

[Pipenv](https://docs.pipenv.org)

For initial setup, run:

```sh
# Install virtualenv in local directory
export PIPENV_VENV_IN_PROJECT=1

pipenv install --dev --three
```

VSCode should automatically load the virtualenv located in the current directory. If not, go to `Python: Select Interpreter` from the command palette.

**Packaging**

[flit](https://flit.readthedocs.io/en/latest/)

**Formatting**

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

**Linting**

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
