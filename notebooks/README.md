# Training on Kaggle

Training runs on Kaggle for the free GPU; HAM10000 is pulled from the Hub. The
local repo handles development, LIME, and version control.

## Setup

1. Notebook **Settings → Accelerator → GPU T4 x2**.
2. Add the Weights & Biases key (<https://wandb.ai/authorize>) under
   **Add-ons → Secrets** as `WANDB_API_KEY`.

## Notebook

```python
# torch ships with Kaggle's GPU image; only install what's missing.
!pip install -q lime wandb -U "transformers>=4.41" "datasets>=2.19"

import os, wandb
from kaggle_secrets import UserSecretsClient
os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
wandb.login()

!git clone https://github.com/haroonmariyam/skin-lesion-xai.git

# Editable installs are unreliable on Kaggle's kernel; add src to the path.
import sys
sys.path.insert(0, "/kaggle/working/skin-lesion-xai/src")

from skin_lesion_xai import train
train.train(model_key="vit", subset_fraction=0.1, epochs=1)  # smoke-test
# train.train(model_key="vit")
# train.train(model_key="resnet")
```

## Exporting the trained models

Models are written to `models/` inside the cloned repo. Zip and download from
the **Output** panel:

```python
import shutil, glob
for key in ["vit", "resnet"]:
    found = glob.glob(f"/kaggle/working/**/models/{key}-ham10000-binary",
                      recursive=True)
    if found:
        shutil.make_archive(f"/kaggle/working/{key}_model", "zip", found[0])

```

Unzip into `models/` locally, then:

```bash
uv run python scripts/03_explain.py --model-dir models/vit-ham10000-binary
```
