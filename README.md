# Chest X-ray Federated Learning (NIH)

## Overview
This repository organizes your **Federated Learning** experiments on the NIH Chest X-ray dataset. 
We preserve your original notebook while providing a reproducible project layout and a script export.

- Original notebook: `notebooks/NIH_Chest_Xray_Federated_Learning.ipynb`
- Exported notebook code: `src/notebook_export.py`
- Minimal launcher: `src/train.py`
- Default config: `configs/default.yaml`

> **Note:** `src/notebook_export.py` concatenates notebook cells in order. For long-term maintenance, consider refactoring into modules such as `datasets.py`, `models.py`, and `train.py`.

## Quickstart
```bash
git clone https://github.com/<YOUR-USER-OR-ORG>/chestxray-federated-learning.git
cd chestxray-federated-learning
bash scripts/run_local.sh
```
Or manually:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python src/train.py --workdir .
```

## Dataset
- The experiments use the NIH Chest X-ray (ChestX-ray14) dataset.
- Download/preprocessing follows the logic embedded in the notebook. (Some cells may rely on Google Colab; adjust or skip those when running locally.)
- If you reorganize the data into an `ImageFolder` layout, update paths in `configs/default.yaml` accordingly.

## Environment
`requirements.txt` was generated based on actual imports detected in the notebook.
- Please install **PyTorch / CUDA** versions appropriate for your machine (e.g., CUDA 12.x).

## Citation
If this code or its results are useful for your research, please cite this repository via `CITATION.cff`.

## License
MIT License (see `LICENSE`)
