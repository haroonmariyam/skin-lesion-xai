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
# 1. Install the few libraries Kaggle doesn't ship by default
!pip install -q lime wandb "transformers>=4.40" "datasets>=2.19"

# 2. Log in to Weights & Biases using the Kaggle secret
from kaggle_secrets import UserSecretsClient
import os, wandb
os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
wandb.login()

# 3. Pull this project's code (push it to GitHub first), then train
!git clone https://github.com/<your-username>/skin-lesion-xai.git
%cd skin-lesion-xai
!pip install -e .

from skin_lesion_xai import train
train.train(model_key="vit")      # then repeat with "resnet"
```

## After training

- Download the saved model folder from the Kaggle notebook's **Output** tab.
- Place it under `models/` on your laptop.
- Run `uv run python scripts/03_explain.py --model-dir models/<name>` locally to
  generate the LIME figures for your report.

> Tip: do one quick `train.train(model_key="vit", subset_fraction=0.1,
> epochs=1)` first to confirm everything runs before a full training job.
