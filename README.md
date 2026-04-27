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
* Symlink or copy datasets:
  `ln -s /home/mwrochna/Prox-LoRA/data/retinopathy/224/ data/retinopathy/224`

## Data preprocessing
### Preprocessed data
Kaggle's [Diabetic Retinopathy](https://www.kaggle.com/competitions/diabetic-retinopathy-detection) dataset is preprocessed (cropped and resized).

To reconstruct the preprocessed datasets, download the original dataset and run `notebooks/preprocess-retinopathy.ipynb`.

### Original dataset
If you want the full original data, you'll need 2 × 83 GB free disk space.
* Create a `KAGGLE_API_TOKEN`: from [Settings](https://www.kaggle.com/settings), under *API Tokens*, click *"Generate New Token"*. Check with `KAGGLE_API_TOKEN=...  uv run kaggle config view`. It should show the right username.
* Click *Verify Phone* (top of Kaggle settings) and then [Join competition](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data) to accept rules.

```bash
mkdir -p data/retinopathy/original
cd data/retinopathy/original
KAGGLE_API_TOKEN=... kaggle competitions download diabetic-retinopathy-detection
7z x -y -sdel sampleSubmission.csv.zip
7z x -y -sdel trainLabels.csv.zip
7z x -y -sdel diabetic-retinopathy-detection.zip
7z x -y train.zip.001 && rm train.zip.*
7z x -y test.zip.001 && rm test.zip.*
wget 'https://storage.googleapis.com/kaggle-forum-message-attachments/90528/2877/retinopathy_solution.csv' -O testLabels.csv
```


## Commands
(You might need to add `--extra cpu` after `uv run` to avoid downloading CUDA).
* Training: `uv run src/prox_lora/train.py`


## Development
* Type checking: `mypy src/` or [Mypy Type Checker](https://marketplace.visualstudio.com/items?itemName=ms-python.mypy-type-checker) extension.
* Linting: `ruff check src/` or [Ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) extension.
* Auto import sorting: `ruff check --select I  --fix` or `Shift+LeftAlt+O`.
* Auto formatting: `ruff format src/` (possibly with `--check`) or `Ctrl+Shift+I`.
