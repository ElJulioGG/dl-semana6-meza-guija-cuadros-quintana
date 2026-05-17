# DL Week 6 — CNN Architectures: Comparison, Batch Normalization and Transfer Learning

Group practical assignment on blood cell classification using classical convolutional neural
networks. LeNet-5 and VGG-11 are implemented from scratch, the effect of Batch Normalization
is analyzed, and Transfer Learning is applied with ResNet-18 pretrained on ImageNet.

**Dataset:** [Blood Cell Image Dataset](https://www.kaggle.com/datasets/paultimothymooney/blood-cells) — 4 classes (EOSINOPHIL, LYMPHOCYTE, MONOCYTE, NEUTROPHIL), ~12,500 RGB images.  
**Framework:** PyTorch  
**Authors:** Meza · Guija · Cuadros · Quintana

---

## Repository Structure

```
dl-semana6-meza-guija-cuadros-quintana/
├── data/
│   ├── archive.zip           # Kaggle dataset
│   └── download_data.sh      # Download script
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory data analysis
│   ├── 02_lenet_vgg.ipynb    # Tasks 1 & 2: LeNet, VGG-11, Batch Normalization
│   └── 03_transfer.ipynb     # Task 3: Transfer Learning with ResNet-18
├── src/
│   ├── models.py             # Architecture definitions
│   ├── train.py              # Training loop
│   └── utils.py              # Utilities
├── results/
│   └── figures/              # Generated plots
├── informe.pdf               # Full technical report
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.10+
- PyTorch (CPU or GPU)

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Installation and Execution

### Mac / Linux

```bash
# 1. Clone the repository
git clone https://github.com/ElJulioGG/dl-semana6-meza-guija-cuadros-quintana
cd dl-semana6-meza-guija-cuadros-quintana

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place the dataset at data/archive.zip (downloaded from Kaggle)
#    or run the download script:
bash data/download_data.sh

# 5. Open Jupyter
python -m notebook
```

### Windows

```powershell
.\setup.ps1
.\venv\Scripts\Activate.ps1
python -m notebook
```

### Google Colab (recommended for GPU)

Open each notebook directly in Colab. In the first cell, uncomment the automatic
download block via Kaggle API.

---

## Execution Order

Run the notebooks in order:

1. `01_eda.ipynb` — Exploratory data analysis
2. `02_lenet_vgg.ipynb` — LeNet-5 and VGG-11 implementation and training (Tasks 1 and 2)
3. `03_transfer.ipynb` — Transfer Learning with ResNet-18 (Task 3)

---

## Main Results

### Tasks 1 & 2 — From-Scratch Models

| Model | Parameters | Time/epoch | Test Accuracy |
|-------|-----------|------------|--------------|
| LeNet-5 | 337,976 | 7.5s | 75.23% |
| LeNet-5 + BN | 338,020 | 7.5s | 66.51% |
| VGG-11s | 2,963,396 | 38.1s | 24.93% |
| VGG-11s + BN | 2,966,148 | 43.6s | **82.51%** |

> Batch Normalization is critical for VGG-11: without it the network fails to converge (24.93%).
> With BN it reaches 82.51%, demonstrating the effect of Internal Covariate Shift (Ioffe & Szegedy, 2015).

### Task 3 — Transfer Learning with ResNet-18

| Strategy | Trainable Params | Test Accuracy | Time/epoch |
|----------|-----------------|--------------|------------|
| Feature Extraction | 2,052 | 58.18% | 206.6s |
| Partial Fine-tuning | 10,495,492 | 79.82% | 335.8s |
| Full Fine-tuning | 11,178,564 | **85.36%** | 460.3s |

> The best overall result is Full Fine-tuning (85.36%). For limited medical data,
> Partial Fine-tuning is recommended for its better balance between performance and overfitting.

---

## Key Findings

- Batch Normalization is indispensable for training VGG-11 from scratch.
- Full Fine-tuning of ResNet-18 outperforms the best from-scratch model by only 2.85 pp.
- Pure Feature Extraction is not suitable for medical microscopy images.
- For small medical datasets, Partial Fine-tuning offers the best trade-off.
