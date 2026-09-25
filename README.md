# ai
This repository contains my experiments, and projects related to Artificial Intelligence and Machine Learning.

## Programming for AI (M2) – CNNs & Transfer Learning on CIFAR-10

| Folder / file | Content |
|---|---|
| `Notebooks/utils.py` | shared code: data loaders, models (`Net`, `CustomCNN`, `get_resnet18`), training loop, evaluation, checkpoints, plots |
| `Notebooks/01_baseline.ipynb` | Task 1 – baseline CNN from scratch |
| `Notebooks/02_experiments.ipynb` | Task 2 – depth / kernel / stride / BN / optimizers / lr / schedulers / augmentation |
| `Notebooks/03_transfer_learning.ipynb` | Task 3 – ResNet-18 frozen vs fine-tuned vs scratch |
| `Notebooks/04_demo_inference.ipynb` | defense – load any checkpoint and test it on test mini-batches |
| `Checkpoints/` | best model of every run (`<run name>.pth`) |
| `results/` | training histories (`<run name>.json`), comparison tables (`.csv`) and `figures/` |
| `Report/main.tex` | IEEE report skeleton |
| `Presentation/outline.md` | slide plan |
| `Docs/defense_notes.md` | checklist, where to make live modifications, theory questions |

### Setup
```bash
pip install -r requirements.txt   # for an NVIDIA GPU, install torch first from https://pytorch.org
cd Notebooks
jupyter notebook
```
CIFAR-10 is downloaded automatically to `data/` on the first run.

### Training on Google Colab (free GPU)
1. Upload the whole `ai` folder to **My Drive** (so the path is `My Drive/ai/Notebooks/...`). The `data/` folder is not needed.
2. Open a notebook from Drive → *Open with → Google Colaboratory*.
3. **Runtime → Change runtime type → T4 GPU**.
4. Run all cells. The first cell mounts Drive (accept the permission popup) and moves into `Notebooks/`.
   If the project is in another Drive folder, change `PROJECT_DIR` in that cell.
5. Checkpoints and results are written to `My Drive/ai/Checkpoints` and `My Drive/ai/results`.
   If Colab disconnects, just run the notebook again: finished runs are skipped.
6. At the end, download `Checkpoints/` and `results/` into the local project for the defense.

Colab already has PyTorch installed, nothing to `pip install`. Keep the browser tab open while training (free Colab disconnects idle sessions).

### Run order
1. `01_baseline.ipynb` (~20 epochs)
2. `02_experiments.ipynb` (20 runs, ~1 h on a GPU)
3. `03_transfer_learning.ipynb` (8 runs, ResNet-18 at 128x128)
4. `04_demo_inference.ipynb`

Each notebook has a `TRAIN` flag in its settings cell: `True` trains (runs that are already saved are skipped), `False` only loads the saved checkpoints and histories so the notebook runs in a few seconds/minutes.
