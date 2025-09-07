# Chest X-ray Federated Learning (NIH)

**한국어 (Korean)** | [English](#english)

## 소개
이 리포지토리는 NIH Chest X-ray 데이터셋을 기반으로 한 **연합학습(Federated Learning)** 실험 코드를 정리한 것입니다. 
업로드하신 주피터 노트북을 그대로 보존하면서, 재현 가능한 실험을 위해 스크립트와 프로젝트 구조를 구성했습니다.

- 원본 노트북: `notebooks/NIH_Chest_Xray_Federated_Learning.ipynb`
- 노트북 코드 내보낸 스크립트: `src/notebook_export.py`
- 실행용 런처(간이): `src/train.py`
- 기본 설정: `configs/default.yaml`

> **참고:** 현재 `src/notebook_export.py`는 노트북의 코드 셀을 순서대로 결합한 파일입니다. 추후 유지보수를 위해 `datasets.py`, `models.py`, `train.py` 등으로 리팩토링하는 것을 권장합니다.

## 빠른 시작
```bash
git clone https://github.com/<YOUR-USER-OR-ORG>/chestxray-federated-learning.git
cd chestxray-federated-learning
bash scripts/run_local.sh
```
또는 수동으로:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python src/train.py --workdir .
```

## 데이터셋
- 본 실험은 NIH Chest X-ray (ChestX-ray14) 데이터셋을 사용합니다. 
- 데이터 다운로드/전처리 방식은 노트북 내부 로직을 따릅니다. (일부 셀은 Google Colab 의존 코드가 포함될 수 있습니다. 로컬 환경에서는 해당 셀을 우회하거나 수정하세요.)
- 폴더 구조가 `ImageFolder` 형식이라면 `configs/default.yaml`의 경로를 조정해 주세요.

## 환경
`requirements.txt`는 노트북에서 실제 사용된 파이썬 패키지를 기준으로 생성했습니다. 
- **PyTorch/CUDA** 버전은 사용 환경에 맞게 설치하세요. (예: CUDA 12.x)

## 인용
이 저장소 또는 결과물을 연구에 사용하셨다면 `CITATION.cff`를 참고해 인용해 주세요.

## 라이선스
MIT License (상세 내용은 `LICENSE` 파일 참조)

---

# English

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
