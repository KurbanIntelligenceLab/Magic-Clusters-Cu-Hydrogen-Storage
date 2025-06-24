# Magic-Clusters-Cu-Hydrogen-Storage

A multi-modal deep learning framework for predicting properties of copper-based hydrogen storage clusters, leveraging graph, image, and tabular data.

---

## Overview

This project aims to predict key physical and chemical properties (such as HOMO, LUMO, band gap, and formation energies) of copper-hydrogen clusters using a multi-modal neural network. The model integrates:

- **Graph-based molecular structure**
- **2D molecular images**
- **Tabular chemical descriptors**

The pipeline includes data preprocessing, knowledge graph construction, data augmentation via molecular rotations, and a flexible training/evaluation loop.

---

## Features

- **Multi-modal Fusion:** Combines graph, image, and tabular data for robust property prediction.
- **Knowledge Graph Construction:** Aggregates molecular properties and relationships from raw data.
- **Automated Data Augmentation:** Generates rotated molecular structures and corresponding images.
- **Leave-One-Cluster-Out Cross-Validation:** Evaluates model generalization to unseen clusters.
- **Modality Contribution Tracking:** Analyzes the importance of each data modality.
- **Configurable Training:** Easily adjust model, training, and data parameters.

---

## Data Preparation

1. **Download the Data:**
   - Download the prepared dataset from the following link:
     
     [Download Data Here](YOUR_DATA_LINK_HERE)

2. **Place Data:**
   - Extract and place the downloaded data into the `new_data/` directory in the project root.

---

## Model Architecture

- **Modified SchNet:** Learns from atomic positions and types (graph input).
- **Modified ResNet50:** Extracts features from 2D molecular images.
- **Tabular GNN:** Processes tabular chemical descriptors with a GNN and residual connections.
- **Fusion Layer:** Attention and gating mechanisms combine all modalities, with auxiliary heads for each.
- **Output:** Predicts target properties (e.g., HOMO, LUMO, Eg, Ef_t, Ef_f).

---

## Training & Evaluation

- **Script:** `train.py`
- **Cross-validation:** Leave specified clusters out for testing, train on the rest.
- **Normalization:** Per-feature scaling for tabular and target properties.
- **Early Stopping & Regularization:** Configurable patience, dropout, and gradient clipping.
- **Results:** Saved in the `results/` directory, including normalization stats and predictions.

---

## Usage

### 1. Set Up Environment & Install Dependencies

It is recommended to use a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Download and Prepare Data

- Download the dataset from the link above and place it in the `new_data/` directory.

### 3. Run Training

```bash
python train.py
```

Model and training parameters can be adjusted in `train.py` under the `CONFIG` dictionary.

---

## Code Quality & Formatting

- **Linting:**
  ```bash
  flake8 .
  ```
- **Auto-formatting:**
  ```bash
  black .
  ```
- **Import Sorting:**
  ```bash
  isort .
  ```

---

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## File Structure

- `model.py` — Defines the multi-modal neural network.
- `train.py` — Training and evaluation pipeline.
- `data_utils.py` — Data loading and preprocessing utilities.
- `new_data/` — Place your downloaded and extracted data here.
- `results/` — Model outputs and logs.

---

## Citation

If you use this code or data, please cite appropriately.
