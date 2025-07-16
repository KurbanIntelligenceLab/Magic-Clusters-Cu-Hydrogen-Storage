import json
import os
from typing import Any, Dict, List

import numpy as np
import torch
from ase.io import read
from PIL import Image
from torch_geometric.data import Data
from torchvision import transforms


def load_knowledge_graph(path: str = "data/knowledge_graph.json") -> Any:
    """Load a knowledge graph from a JSON file."""
    with open(path) as f:
        return json.load(f)


def load_xyz_as_pyg_data(xyz_path: str) -> Data:
    """Load an XYZ file and return a PyTorch Geometric Data object."""
    atoms = read(xyz_path)
    pos: torch.Tensor = torch.tensor(atoms.get_positions(), dtype=torch.float)
    atomic_numbers: torch.Tensor = torch.tensor(
        atoms.get_atomic_numbers(), dtype=torch.long
    )
    data = Data(z=atomic_numbers, pos=pos)
    return data


def load_image(image_path: str) -> torch.Tensor:
    """Load an image, apply standard transforms, and return a tensor."""
    img: Image.Image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return transform(img)


def build_tabular_tensor(node: Dict[str, Any], tabular_keys: List[str]) -> torch.Tensor:
    """Build a tensor from a node dictionary and a list of tabular keys."""
    return torch.tensor([node.get(k, 0.0) for k in tabular_keys], dtype=torch.float)


def load_results():
    """Load and average results from up to num_seeds seeds, structures, and properties."""
    results_dir = "results"
    all_results = {}
    seed_dirs = [d for d in os.listdir(results_dir) if d.startswith("seed_")]
    seed_dirs = sorted(seed_dirs)
    if not seed_dirs:
        return all_results

    # Get all structures and properties from the first seed
    first_seed = seed_dirs[0]
    structure_dirs = [
        d
        for d in os.listdir(os.path.join(results_dir, first_seed))
        if d.startswith("R")
    ]
    for structure in structure_dirs:
        all_results[structure] = {}
        property_dirs = [
            d
            for d in os.listdir(os.path.join(results_dir, first_seed, structure))
            if not d.startswith(".")
        ]
        for prop in property_dirs:
            preds, trues, maes = [], [], []
            for seed in seed_dirs:
                result_file = os.path.join(
                    results_dir, seed, structure, prop, "results.json"
                )
                if os.path.exists(result_file):
                    with open(result_file, "r") as f:
                        data = json.load(f)
                        preds.append(np.array(data["pred"]))
                        trues.append(np.array(data["true"]))
                        maes.append(data["mae"])
            if preds:
                # Average across seeds
                avg_pred = np.mean(preds, axis=0).tolist()
                avg_true = np.mean(trues, axis=0).tolist()
                avg_mae = float(np.mean(maes))
                all_results[structure][prop] = {
                    "pred": avg_pred,
                    "true": avg_true,
                    "mae": avg_mae,
                }
    return all_results
