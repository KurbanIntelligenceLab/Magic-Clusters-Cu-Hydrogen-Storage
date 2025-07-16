# Magic-Clusters-Cu-Hydrogen-Storage

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

### 3. Prepare Data

- Place your dataset in the `new_data/` directory as required by the scripts.
- The folder structure should look like:
  ```
  Magic-Clusters-Cu-Hydrogen-Storage/
    ├── new_data/
    │   ├── initial-strcutures/
    │   ├── optimized-structures/
    │   ├── optimized-structures-with-H2/
    │   ├── energies.txt
    │   └── Formation_energy-EF.txt
    ├── utils/
    ├── model.py
    ├── train.py
    ├── config.py
    └── ...
  ```

### 4. Train the Model

```bash
python train.py
```

- Training results and logs will be saved in the `results/` directory.
- **You can modify model and training parameters in `config.py` to suit your experiments.**
- Make sure you modify the output layer of SchNet from 1 to `schnet_out`.
---

For any questions or issues, please open an issue on the repository.
