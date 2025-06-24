import json
from typing import Any, Dict, List

import torch
from ase.io import read
from PIL import Image
from torch_geometric.data import Data
from torchvision import transforms


def load_knowledge_graph(path: str) -> Any:
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
