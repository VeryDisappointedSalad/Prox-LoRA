# ProxLoRA


## Installation
* `git clone git@github.com:VeryDisappointedSalad/Prox-LoRA.git`
* `cd Prox-LoRA`
* Install [uv](https://docs.astral.sh/uv/getting-started/installation/):
  `curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc`
* Setup the venv (~8GiB without CUDA), either:
    * `uv sync --extra cpu`
    * `uv sync --extra cu128` (if using GPUs compatible with CUDA 12.8)
* `cp .env.template .env` and edit `.env`:
  * `CLEARML_API_ACCESS_KEY` and `CLEARML_API_SECRET_KEY`:
    [create new credentials](https://app.clear.ml/settings/workspace-configuration) for the right workspace
    and copy values shown (only) during creation.
    Check with `UV_ENV_FILE=.env uv run clearml-debug token` . It should show the right 'company_name' and 'user_name', not an error.


## Commands
* Training: `uv run src/prox_lora/train.py`


## Development
* Type checking: `mypy src/` or [Mypy Type Checker](https://marketplace.visualstudio.com/items?itemName=ms-python.mypy-type-checker) extension.
* Linting: `ruff check src/` or [Ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) extension.
* Auto import sorting: `ruff check --select I  --fix` or `Shift+LeftAlt+O`.
* Auto formatting: `ruff format src/` (possibly with `--check`) or `Ctrl+Shift+I`.

