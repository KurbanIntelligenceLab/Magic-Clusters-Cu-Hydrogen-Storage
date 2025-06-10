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
    energies_df = pd.read_csv('data/energies.txt', sep=r'\s+')
    formation_df = pd.read_csv('data/formation_energies.txt', sep=r'\s+')
    material_mapping = {
        'Cu_R7_optimized': {'energies_key': 'R0.7', 'formation_key': 'R7', 'radius': 0.7},
        'Cu_R8_optimized': {'energies_key': 'R0.8', 'formation_key': 'R8', 'radius': 0.8},
        'Cu_R9_optimized': {'energies_key': 'R0.9', 'formation_key': 'R9', 'radius': 0.9},
        'Cu_R10_optimized': {'energies_key': 'R1.0', 'formation_key': 'R10', 'radius': 1.0},
    }
    
    G = nx.Graph()
    
    # Add material nodes with their properties
    for material, keys in material_mapping.items():
        # Get energy features for this specific material
        energies_features = energies_df.loc[energies_df.index == keys['energies_key']].iloc[0].to_dict()
        formation_features = formation_df.loc[formation_df.index == keys['formation_key']].iloc[0].to_dict()
        
        # Add radius information
        material_features = {
            'radius': keys['radius'],
            **energies_features,
            **{f'formation_{k}': v for k, v in formation_features.items()}
        }
        
        # Add node for the material
        G.add_node(material, features=material_features, type='material')
    
    # Add edges between materials based on multiple criteria
    materials = list(material_mapping.keys())
    for i in range(len(materials)):
        for j in range(i + 1, len(materials)):
            mat1, mat2 = materials[i], materials[j]
            
            # Calculate multiple similarity metrics
            radius1 = material_mapping[mat1]['radius']
            radius2 = material_mapping[mat2]['radius']
            radius_diff = abs(radius1 - radius2)
            
            # Get energy features for similarity calculation
            energy1 = energies_df.loc[energies_df.index == material_mapping[mat1]['energies_key']].iloc[0]
            energy2 = energies_df.loc[energies_df.index == material_mapping[mat2]['energies_key']].iloc[0]
            energy_similarity = 1.0 - np.mean(np.abs(energy1 - energy2) / (np.abs(energy1) + np.abs(energy2) + 1e-8))
            
            # Combine similarities
            if radius_diff <= 0.3:  # Only connect relatively similar materials
                radius_similarity = 1.0 - (radius_diff / 0.3)
                total_similarity = 0.5 * radius_similarity + 0.5 * energy_similarity
                G.add_edge(mat1, mat2, weight=total_similarity, 
                          radius_similarity=radius_similarity,
                          energy_similarity=energy_similarity)
    
    print("Knowledge graph built:")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    
    # Function to compute structural features from XYZ data
    def compute_structural_features(xyz_data):
        coords = np.array([coord for _, coord in xyz_data])
        features = {}
        
        # Compute pairwise distances
        distances = np.linalg.norm(coords[:, np.newaxis] - coords, axis=2)
        np.fill_diagonal(distances, np.inf)  # Ignore self-distances
        
        # Basic structural features
        features['min_distance'] = np.min(distances)
        features['max_distance'] = np.max(distances)
        features['mean_distance'] = np.mean(distances)
        features['std_distance'] = np.std(distances)
        
        # Coordination number (atoms within 3.0 Å)
        coordination = np.sum(distances < 3.0, axis=1)
        features['min_coordination'] = np.min(coordination)
        features['max_coordination'] = np.max(coordination)
        features['mean_coordination'] = np.mean(coordination)
        
        # Density features
        volume = np.prod(np.max(coords, axis=0) - np.min(coords, axis=0))
        features['density'] = len(coords) / volume if volume > 0 else 0
        
        # Symmetry features
        center = np.mean(coords, axis=0)
        radial_distances = np.linalg.norm(coords - center, axis=1)
        features['radial_std'] = np.std(radial_distances)
        features['radial_mean'] = np.mean(radial_distances)
        
        return features
    
    # Function to get neighborhood features
    def get_neighborhood_features(graph, material):
        """Get features of the material and its neighbors."""
        features = graph.nodes[material]['features'].copy()
        
        # Add neighborhood information
        neighbors = list(graph.neighbors(material))
        if neighbors:
            neighbor_features = {
                'num_neighbors': len(neighbors),
                'avg_neighbor_radius': np.mean([graph.nodes[n]['features']['radius'] for n in neighbors]),
                'min_neighbor_radius': min(graph.nodes[n]['features']['radius'] for n in neighbors),
                'max_neighbor_radius': max(graph.nodes[n]['features']['radius'] for n in neighbors),
                'avg_edge_weight': np.mean([graph[material][n]['weight'] for n in neighbors]),
                'avg_energy_similarity': np.mean([graph[material][n]['energy_similarity'] for n in neighbors]),
                'avg_radius_similarity': np.mean([graph[material][n]['radius_similarity'] for n in neighbors])
            }
        else:
            neighbor_features = {
                'num_neighbors': 0,
                'avg_neighbor_radius': 0,
                'min_neighbor_radius': 0,
                'max_neighbor_radius': 0,
                'avg_edge_weight': 0,
                'avg_energy_similarity': 0,
                'avg_radius_similarity': 0
            }
        
        features.update(neighbor_features)
        return features

    # Modify get_sample_data to use enhanced features
    def get_sample_data(graph, material, rotation):
        if not graph.has_node(material):
            raise ValueError(f"Material node '{material}' not found in the graph.")
        
        # Get features including neighborhood information
        material_features = get_neighborhood_features(graph, material)
        
        file_label = "original" if rotation == 0 else f"rot_{rotation}"
        xyz_file = f"data/multimodal_data/{material}/xyzs/{file_label}.xyz"
        image_file = f"data/multimodal_data/{material}/images/{file_label}.png"
        xyz_data = load_xyz_file(xyz_file)
        
        # Add structural features
        structural_features = compute_structural_features(xyz_data)
        material_features.update(structural_features)
        
        # Add rotation-specific features
        rotation_features = {
            'rotation_angle': rotation,
            'is_original': 1.0 if rotation == 0 else 0.0
        }
        material_features.update(rotation_features)
        
        return material_features, xyz_data, image_file

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
            
            # Check and store feature ranges
            self.feature_min = {}
            self.feature_max = {}
            for k in self.feature_keys:
                min_val = np.min(all_features[k])
                max_val = np.max(all_features[k])
                if np.isclose(min_val, max_val):
                    print(f"Warning: Feature '{k}' has identical values across all samples: {min_val}")
                self.feature_min[k] = min_val
                self.feature_max[k] = max_val
            
            # Check and store label range
            all_labels = list(label_mapping.values())
            self.label_min = np.min(all_labels)
            self.label_max = np.max(all_labels)
            if np.isclose(self.label_min, self.label_max):
                raise ValueError(f"All labels have identical value: {self.label_min}. This indicates a problem with the dataset.")
            
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            material, rotation = self.samples[idx]
            features, xyz_data, image_file = get_sample_data(self.graph, material, rotation)
            normalized_features = []
            for k in self.feature_keys:
                min_val = self.feature_min[k]
                max_val = self.feature_max[k]
                # Add small epsilon to denominator to prevent division by zero
                norm = (features[k] - min_val) / (max_val - min_val + 1e-8)
                normalized_features.append(norm)
            
            features_tensor = torch.tensor(normalized_features, dtype=torch.float)
            image = Image.open(image_file).convert('RGB')
            if self.image_transform:
                image = self.image_transform(image)
            
            formation_key = self.material_mapping[material]['formation_key']
            label_value = self.label_mapping[formation_key]
            # Add small epsilon to denominator for labels as well
            normalized_label = (label_value - self.label_min) / (self.label_max - self.label_min + 1e-8)
            
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
