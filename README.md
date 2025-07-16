# Multimodal Explainable Artificial Intelligence–Driven Analysis of Quantum Size Effects in Copper Nanoclusters for Hydrogen Storage

## Authors

- **Mustafa Kurban** - Department of Prosthetics and Orthotics, Ankara University, Ankara, Turkey & Department of Electrical and Computer Engineering, Texas A&M University at Qatar, Doha, Qatar
- **Can Polat** - Department of Electrical and Computer Engineering, Texas A&M University, College Station, TX, USA
- **Erchin Serpedin** - Department of Electrical and Computer Engineering, Texas A&M University, College Station, TX, USA
- **Hasan Kurban** - College of Science and Engineering, Hamad Bin Khalifa University, Doha, Qatar

## Getting Started

### 1. Clone the Repository

```bash
git clone git@github.com:KurbanIntelligenceLab/Magic-Clusters-Cu-Hydrogen-Storage.git
cd Magic-Clusters-Cu-Hydrogen-Storage
```

### 2. Install Dependencies

It's recommended to use a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Data Structure

The project expects the following data structure:

```
Magic-Clusters-Cu-Hydrogen-Storage/
├── data/
│   ├── knowledge_graph.json          # Knowledge graph with node relationships
│   ├── labels.csv                    # Target labels for training
│   ├── rotated_optimized-structures/
│   │   ├── images/                   # PNG images of molecular structures
│   │   └── xyz_files/               # XYZ coordinate files
│   └── rotated_optimized-structures-with-H2/
│       ├── images/                   # PNG images with H2 molecules
│       └── xyz_files/               # XYZ files with H2 molecules
├── utils/
│   ├── data_utils.py                # Data loading and processing utilities
│   ├── create_knowledge_graph.py    # Knowledge graph creation
│   └── generate_rotations.py        # Molecular rotation generation
├── model.py                         # MultiModalModel with SchNet, ResNet, and GNN
├── train.py                         # Training script with multimodal fusion
├── config.py                        # Configuration parameters
└── requirements.txt                 # Python dependencies
```

### 4. Model Architecture

The project uses a multimodal architecture combining:

- **SchNet**: Geometric neural network for molecular structure analysis
- **ResNet-50**: Image processing for molecular visualizations  
- **Tabular GNN**: Graph neural network for tabular feature processing
- **Attention-based Fusion**: Combines all modalities with learned weights

### 5. Train the Model

```bash
python train.py
```

Training results and logs will be saved in the `results/` directory.

### 6. Important Note: SchNet Modification

Before training, you need to modify the SchNet implementation in your PyTorch Geometric installation. In the file `venv/lib/python3.12/site-packages/torch_geometric/nn/models/schnet.py`, change line 148:

```python
# Original line:
self.lin2 = Linear(hidden_channels // 2, 1)

# Modified line:
self.lin2 = Linear(hidden_channels // 2, output_dim)
```

This ensures SchNet outputs the correct dimension (`schnet_out`) as specified in your model configuration, rather than the default output dimension of 1.

### 7. Configuration

You can modify model and training parameters in `config.py`:

- **Model parameters**: Architecture dimensions, fusion settings
- **Training parameters**: Learning rate, batch size, epochs
- **Data parameters**: Target properties, tabular features
- **Regularization**: Dropout, early stopping, gradient clipping

### 8. Key Features

- **Multimodal Learning**: Combines geometric, image, and tabular data
- **Knowledge Graph Integration**: Uses molecular relationships for enhanced predictions
- **Rotation Augmentation**: Multiple molecular orientations for robustness
- **Attention Mechanisms**: Learns optimal modality fusion weights
- **Auxiliary Heads**: Individual modality predictions for interpretability

### 9. Dependencies

The project requires:
- PyTorch and PyTorch Geometric for deep learning
- ASE for molecular structure handling
- OpenCV and PIL for image processing
- NetworkX for graph operations
- Standard ML libraries (scikit-learn, numpy, matplotlib)

For any questions or issues, please open an issue on the repository.
