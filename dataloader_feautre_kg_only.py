import networkx as nx
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from torchvision import transforms
import numpy as np
import random
from sklearn.model_selection import KFold  # new import

def create_data_generator(material_to_test, seed, num_of_folds=5, batch_size=32):
    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
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

    # -------------------------------
    # 1. Build the Knowledge Graph
    # -------------------------------
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
    print("Knowledge graph built:")
    print("Number of nodes:", G.number_of_nodes())

    # -------------------------------
    # 2. Load Labels
    # -------------------------------
    labels_df = pd.read_csv('data/labels.txt', sep=r'\s+', header=None, index_col=0)
    labels_df.columns = ['label']
    label_mapping = labels_df['label'].to_dict()

    # -------------------------------
    # 3. Helper Functions
    # -------------------------------
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
        if not graph.has_node(material):
            raise ValueError(f"Material node '{material}' not found in the graph.")
        material_features = graph.nodes[material]['features']
        file_label = "original" if rotation == 0 else f"rot_{rotation}"
        xyz_file = f"data/multimodal_data/{material}/xyzs/{file_label}.xyz"
        image_file = f"data/multimodal_data/{material}/images/{file_label}.png"
        xyz_data = load_xyz_file(xyz_file)
        return material_features, xyz_data, image_file

    # -------------------------------
    # 4. Define Custom Dataset
    # -------------------------------
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
            
            self.feature_keys = sorted(list(graph.nodes[material_list[0]]['features'].keys()))
            all_features = {k: [] for k in self.feature_keys}
            for material in material_list:
                feats = graph.nodes[material]['features']
                for k in self.feature_keys:
                    all_features[k].append(feats[k])
            self.feature_min = {k: np.min(all_features[k]) for k in self.feature_keys}
            self.feature_max = {k: np.max(all_features[k]) for k in self.feature_keys}
            all_labels = list(label_mapping.values())
            self.label_min = np.min(all_labels)
            self.label_max = np.max(all_labels)
            
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            material, rotation = self.samples[idx]
            features, xyz_data, image_file = get_sample_data(self.graph, material, rotation)
            normalized_features = []
            for k in self.feature_keys:
                min_val = self.feature_min[k]
                max_val = self.feature_max[k]
                if max_val > min_val:
                    norm = 2 * ((features[k] - min_val) / (max_val - min_val)) - 1
                else:
                    norm = 0.0
                normalized_features.append(norm)
            features_tensor = torch.tensor(normalized_features, dtype=torch.float)
            image = Image.open(image_file).convert('RGB')
            if self.image_transform:
                image = self.image_transform(image)
            formation_key = self.material_mapping[material]['formation_key']
            label_value = self.label_mapping[formation_key]
            if self.label_max > self.label_min:
                normalized_label = 2 * ((label_value - self.label_min) / (self.label_max - self.label_min)) - 1
            else:
                normalized_label = 0.0
            sample = {
                'material': material,
                'features': features_tensor,
                'xyz_data': xyz_data,
                'image': image,
                'label': normalized_label
            }
            return sample

    # -------------------------------
    # 5. Custom Collate Function
    # -------------------------------
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

    # -------------------------------
    # 6. Instantiate Dataset and DataLoaders
    # -------------------------------
    material_list = list(material_mapping.keys())
    num_rotations = 526

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Reserve one material for testing; use the rest for training/validation
    all_materials = list(material_mapping.keys())
    test_material = material_to_test
    train_val_materials = [mat for mat in all_materials if mat != test_material]
    train_val_dataset = MaterialGraphDataset(G, train_val_materials, material_mapping, label_mapping, num_rotations, image_transform=transform)

    fold_loaders = []
    kf = KFold(n_splits=num_of_folds, shuffle=True, random_state=seed)
    indices = list(range(len(train_val_dataset)))
    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        train_subset = Subset(train_val_dataset, train_idx)
        val_subset = Subset(train_val_dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn,
                                  worker_init_fn=seed_worker, generator=data_gen)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn,
                                  worker_init_fn=seed_worker)
        fold_loaders.append((train_loader, val_loader))
        print(f"Fold {fold + 1}: Train samples: {len(train_idx)}, Validation samples: {len(val_idx)}")

    # Create test dataset and DataLoader (unchanged)
    test_dataset = MaterialGraphDataset(G, [test_material], material_mapping, label_mapping, num_rotations, image_transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn,
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
    return fold_loaders, test_loader
