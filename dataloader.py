import networkx as nx
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import numpy as np
from torch.utils.data import random_split

# --------------------------------------------------
# 1. Build the Knowledge Graph
# --------------------------------------------------
# set all the seeds
torch.manual_seed(42)
np.random.seed(42)
import random
import numpy as np
import torch
import pandas as pd
import networkx as nx
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

# -------------------------
# Reproducibility Settings
# -------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
# Ensure deterministic behavior for CUDNN (may affect performance)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Function to initialize worker seed
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# Create a generator for DataLoader and random_split
data_gen = torch.Generator()
data_gen.manual_seed(seed)

# -------------------------
# 1. Build the Knowledge Graph
# -------------------------
energies_df = pd.read_csv('data/energies.txt', sep=r'\s+', index_col=0)
formation_df = pd.read_csv('data/formation_energies.txt', sep=r'\s+', index_col=0)

material_mapping = {
    'Cu_R7_optimized': {'energies_key': 'R0.7', 'formation_key': 'R7'},
    'Cu_R8_optimized': {'energies_key': 'R0.8', 'formation_key': 'R8'},
    'Cu_R9_optimized': {'energies_key': 'R0.9', 'formation_key': 'R9'},
    'Cu_R10_optimized': {'energies_key': 'R1.0', 'formation_key': 'R10'},
}

G = nx.Graph()
for material, keys in material_mapping.items():
    energies_features = energies_df.loc[keys['energies_key']].to_dict()
    formation_features = {f'formation_{k}': v for k, v in formation_df.loc[keys['formation_key']].to_dict().items()}
    combined_features = {**energies_features, **formation_features}
    G.add_node(material, features=combined_features, type='material')

num_rotations = 526
for material in material_mapping.keys():
    for rot in range(num_rotations):
        file_label = "original" if rot == 0 else f"rot_{rot}"
        xyz_node = f"{material}_rot_{rot}"
        file_path_xyz = f"data/multimodal_data/{material}/xyzs/{file_label}.xyz"
        G.add_node(xyz_node, file=file_path_xyz, modality='xyz')
        G.add_edge(material, xyz_node, relation='has_xyz')
        image_node = f"{material}_img_{rot}"
        image_path = f"data/multimodal_data/{material}/images/{file_label}.png"
        G.add_node(image_node, file=image_path, modality='image')
        G.add_edge(material, image_node, relation='has_image')

print("Knowledge graph built:")
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

# --------------------------------------------------
# 2. Load Labels from labels.txt
# --------------------------------------------------
labels_df = pd.read_csv('data/labels.txt', sep=r'\s+', header=None, index_col=0)
labels_df.columns = ['label']
label_mapping = labels_df['label'].to_dict()

# --------------------------------------------------
# 3. Helper Functions for Data Loading
# --------------------------------------------------
def load_xyz_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    xyz_data = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) >= 4:
            atom = parts[0]
            coords = [float(x) for x in parts[1:4]]
            xyz_data.append((atom, coords))
    return xyz_data

def get_sample_data(graph, material, rotation):
    xyz_node = f"{material}_rot_{rotation}"
    image_node = f"{material}_img_{rotation}"
    if not graph.has_node(material):
        raise ValueError(f"Material node '{material}' not found in the graph.")
    if not graph.has_node(xyz_node):
        raise ValueError(f"XYZ node '{xyz_node}' not found in the graph.")
    if not graph.has_node(image_node):
        raise ValueError(f"Image node '{image_node}' not found in the graph.")
    material_features = graph.nodes[material]['features']
    xyz_file = graph.nodes[xyz_node]['file']
    image_file = graph.nodes[image_node]['file']
    xyz_data = load_xyz_file(xyz_file)
    return material_features, xyz_data, image_file

# --------------------------------------------------
# 4. Define the Custom Dataset with Min-Max Normalization
# --------------------------------------------------
class MaterialGraphDataset(Dataset):
    def __init__(self, graph, material_list, material_mapping, label_mapping, num_rotations, image_transform=None):
        self.graph = graph
        self.samples = []
        self.image_transform = image_transform
        self.material_mapping = material_mapping
        self.label_mapping = label_mapping
        
        for material in material_list:
            for rotation in range(num_rotations):
                self.samples.append((material, rotation))
        
        # Get a consistent ordering for feature keys
        self.feature_keys = sorted(list(graph.nodes[material_list[0]]['features'].keys()))
        
        # Compute min and max for each feature across materials
        all_features = {k: [] for k in self.feature_keys}
        for material in material_list:
            feats = graph.nodes[material]['features']
            for k in self.feature_keys:
                all_features[k].append(feats[k])
        self.feature_min = {k: np.min(all_features[k]) for k in self.feature_keys}
        self.feature_max = {k: np.max(all_features[k]) for k in self.feature_keys}
        
        # Compute min and max for labels
        all_labels = list(label_mapping.values())
        self.label_min = np.min(all_labels)
        self.label_max = np.max(all_labels)
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        material, rotation = self.samples[idx]
        features, xyz_data, image_file = get_sample_data(self.graph, material, rotation)
        
        # Normalize features using min-max scaling to [-1, 1]
        normalized_features = []
        for k in self.feature_keys:
            min_val = self.feature_min[k]
            max_val = self.feature_max[k]
            # Avoid division by zero if max equals min
            if max_val > min_val:
                norm = 2 * ((features[k] - min_val) / (max_val - min_val)) - 1
            else:
                norm = 0.0
            normalized_features.append(norm)
        features_tensor = torch.tensor(normalized_features, dtype=torch.float)
        
        # Load and transform image
        image = Image.open(image_file).convert('RGB')
        if self.image_transform:
            image = self.image_transform(image)
        
        # Normalize label using min-max scaling to [-1, 1]
        formation_key = self.material_mapping[material]['formation_key']
        label_value = self.label_mapping[formation_key]
        if self.label_max > self.label_min:
            normalized_label = 2 * ((label_value - self.label_min) / (self.label_max - self.label_min)) - 1
        else:
            normalized_label = 0.0
        
        sample = {
            'material': material,
            'features': features_tensor,
            'xyz_data': xyz_data,  # list of (atom, [x, y, z]) tuples
            'image': image,
            'label': normalized_label
        }
        return sample

# --------------------------------------------------
# 5. Custom Collate Function for Padding xyz Data
# --------------------------------------------------
def custom_collate_fn(batch):
    materials = []
    features_list = []
    xyz_data_list = []
    xyz_lengths = []
    images = []
    labels = []
    
    for sample in batch:
        materials.append(sample['material'])
        features_list.append(sample['features'])
        coords = torch.tensor([coords for atom, coords in sample['xyz_data']], dtype=torch.float)
        xyz_data_list.append(coords)
        xyz_lengths.append(coords.shape[0])
        images.append(sample['image'])
        labels.append(torch.tensor(sample['label'], dtype=torch.float))
    
    max_atoms = max(xyz_lengths)
    padded_xyz = []
    for coords in xyz_data_list:
        pad_size = max_atoms - coords.shape[0]
        if pad_size > 0:
            pad = torch.zeros((pad_size, 3), dtype=torch.float)
            padded = torch.cat([coords, pad], dim=0)
        else:
            padded = coords
        padded_xyz.append(padded)
    padded_xyz = torch.stack(padded_xyz)
    
    features_tensor = torch.stack(features_list)
    images_tensor = torch.stack(images)
    labels_tensor = torch.stack(labels)
    
    return {
        'material': materials,
        'features': features_tensor,
        'xyz_data': padded_xyz,
        'xyz_lengths': torch.tensor(xyz_lengths),
        'image': images_tensor,
        'label': labels_tensor
    }

# --------------------------------------------------
# 6. Instantiate Dataset and DataLoader
# --------------------------------------------------
material_list = list(material_mapping.keys())
num_rotations = 526

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# All available material folders (from material_mapping keys)
all_materials = list(material_mapping.keys())

# Reserve one material for testing
test_material = 'Cu_R7_optimized'  # change as needed
train_val_materials = [mat for mat in all_materials if mat != test_material]

# Create the combined train+validation dataset
train_val_dataset = MaterialGraphDataset(G, train_val_materials, material_mapping, label_mapping, num_rotations, image_transform=transform)

# Use random_split to divide train_val_dataset into 80% training and 20% validation subsets
total_samples = len(train_val_dataset)
train_size = int(0.8 * total_samples)
val_size = total_samples - train_size
train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size], generator=data_gen)

# Create the test dataset separately using the test material
test_dataset  = MaterialGraphDataset(G, [test_material], material_mapping, label_mapping, num_rotations, image_transform=transform)

# Create corresponding DataLoaders with the worker_init_fn and generator for reproducibility
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn,
                          worker_init_fn=seed_worker, generator=data_gen)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn,
                          worker_init_fn=seed_worker)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn,
                          worker_init_fn=seed_worker)

# for batch in train_loader:
#     # print("Train batch materials:", batch['material'])
#     print("Train batch features shape:", batch['features'].shape)
#     print("Train batch xyz_data shape:", batch['xyz_data'].shape)
#     print("Train batch image shape:", batch['image'].shape)
#     break

# for batch in val_loader:
#     # print("Validation batch materials:", batch['material'])
#     print("Validation batch features shape:", batch['features'].shape)
#     print("Validation batch xyz_data shape:", batch['xyz_data'].shape)
#     print("Validation batch image shape:", batch['image'].shape)
#     break

# for batch in test_loader:
#     # print("Test batch materials:", batch['material'])
#     print("Test batch features shape:", batch['features'].shape)
#     print("Test batch xyz_data shape:", batch['xyz_data'].shape)
#     print("Test batch image shape:", batch['image'].shape)
#     break