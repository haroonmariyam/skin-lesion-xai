# Training on Kaggle

Training deep image models needs a GPU. Kaggle gives you one for free (~30
hrs/week) and already hosts HAM10000, so this is where the heavy `02_train.py`
step should run. Your laptop stays the place where you write code, run LIME on
a few images, and keep the Git history.

## One-time setup

1. Create a free account at <https://www.kaggle.com>.
2. In a new notebook, open **Settings → Accelerator → GPU T4 x2**.
3. Get a Weights & Biases key from <https://wandb.ai/authorize>. In the Kaggle
   notebook, add it under **Add-ons → Secrets** as `WANDB_API_KEY`.

## In the Kaggle notebook

```python
# 1. Install only the libraries Kaggle lacks. We do NOT reinstall torch:
#    Kaggle already ships a GPU build, and replacing it can break the GPU.
!pip install -q lime wandb -U "transformers>=4.41" "datasets>=2.19"

# 2. Log in to Weights & Biases using the Kaggle secret
from kaggle_secrets import UserSecretsClient
import os, wandb
os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
wandb.login()

# 3. Pull this project's code
!git clone https://github.com/haroonmariyam/skin-lesion-xai.git

# 4. Make the package importable. Pointing Python at the source folder is the
#    most reliable approach on Kaggle (editable pip installs can fail to
#    register on the kernel's import path).
import sys
sys.path.insert(0, "/kaggle/working/skin-lesion-xai/src")

# 5. Smoke-test first (10% of data, 1 epoch) to confirm everything runs:
from skin_lesion_xai import train
train.train(model_key="vit", subset_fraction=0.1, epochs=1)

# 5. Once that works, do the full runs (one per model):
# train.train(model_key="vit")
# train.train(model_key="resnet")
```

## After training

- Download the saved model folder from the Kaggle notebook's **Output** tab.
- Place it under `models/` on your laptop.
- Run `uv run python scripts/03_explain.py --model-dir models/<name>` locally to
  generate the LIME figures for your report.

> Tip: do one quick `train.train(model_key="vit", subset_fraction=0.1,
> epochs=1)` first to confirm everything runs before a full training job.
