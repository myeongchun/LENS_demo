<h1 align="center">LENS</h1>

<h3 align="center">
  Learning an Ensemble of Normality Subspaces<br>
  for Unsupervised Anomaly Detection
</h3>

<p align="center">
  <img src="https://img.shields.io/badge/WSDM-2027-2563eb?style=flat-square" alt="WSDM 2027">
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10">
  <img src="https://img.shields.io/badge/PyTorch-2.6-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch 2.6">
</p>

<p align="center">
  Official implementation of <b>LENS</b>, a boosting-based ensemble framework<br>
  for learning complementary normality subspaces.
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#datasets">Datasets</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a>
</p>

---

## Overview

Dimensionality reduction inevitably loses part of the input information.
**LENS** recovers normality missed by preceding models by constructing
a sequence of complementary PCA latent spaces.

LENS regulates reconstruction-error-driven boosting through two mechanisms:

- **Contribution-aware filtering** excludes data points with disproportionately
  high latent contribution scores from subsequent training.
- **Reliability-aware weighting** downweights data points whose latent
  contribution scores vary substantially across consecutive spaces.

By combining these mechanisms, LENS learns diverse aspects of normality
while limiting the influence of dominant or inconsistent data points.

<!--
Upload the framework figure to assets/framework.png, then uncomment:

<p align="center">
  <img src="assets/framework.png" width="900" alt="Overview of the LENS framework">
</p>
-->

## Datasets

LENS is evaluated on **57 benchmark datasets** from
[ADBench](https://github.com/Minqi824/ADBench).

| Data type | Number of datasets | Representation |
| :--- | ---: | :--- |
| Tabular | 47 | Tabular features |
| Image | 5 | ResNet18 embeddings |
| Text | 5 | BERT embeddings |

Download the datasets and place the `.npz` files in the
[`datasets/`](datasets/) directory.

## Installation

Clone the repository:

```bash
git clone https://github.com/myeongchun/LENS.git
cd LENS
```

Use **Python 3.10.19** and install the dependencies:

```bash
pip install -r requirements.txt
```

<details>
<summary><b>Dependency versions</b></summary>

```txt
torch==2.6.0
numpy==1.26.4
scipy==1.15.3
scikit-learn==1.7.2
tqdm==4.67.1
psutil==7.2.1
openpyxl==3.1.5
```

</details>

## Quick Start

Replace `"dataset name"` with the name of an NPZ file in `datasets/`,
with or without the `.npz` extension.

### Run a single dataset

```bash
# GPU 0
python main.py --dataset "dataset name" --device 0

# CPU
python main.py --dataset "dataset name" --device cpu

# Automatically select GPU when available, otherwise CPU
python main.py --dataset "dataset name" --device auto
```

### Run all datasets

```bash
python main.py --dataset all --device 0
```

### Set the random seed

```bash
python main.py --dataset "dataset name" --device 0 --seed 1
```

### Use a configuration file

The default settings are provided in [`config.json`](config.json).
To use another configuration file:

```bash
python main.py --dataset "dataset name" --device 0 --config config.json
```

Each run reports **AUROC** and **AUPRC** as percentages.
