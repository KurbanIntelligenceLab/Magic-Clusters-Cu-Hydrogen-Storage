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

    def analyze_knowledge_graph(G):
        """Analyze and explain the knowledge graph structure."""
        analysis = {
            'basic_stats': {
                'num_nodes': G.number_of_nodes(),
                'num_edges': G.number_of_edges(),
                'density': nx.density(G),
                'avg_clustering': nx.average_clustering(G),
                'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes()
            },
            'material_properties': {},
            'similarity_metrics': {
                'radius_similarity': [],
                'energy_similarity': [],
                'structural_similarity': [],
                'symmetry_similarity': []
            },
            'central_materials': [],
            'community_structure': []
        }
        
        # Analyze material properties
        for node in G.nodes():
            if G.nodes[node]['type'] == 'material':
                features = G.nodes[node]['features']
                analysis['material_properties'][node] = {
                    'radius': features['radius'],
                    'density': features['density'],
                    'radial_symmetry': features['radial_symmetry'],
                    'coordination_number': features['coordination_number']
                }
        
        # Analyze edge similarities
        for u, v, data in G.edges(data=True):
            for metric in ['radius_similarity', 'energy_similarity', 'structural_similarity', 'symmetry_similarity']:
                if metric in data:
                    analysis['similarity_metrics'][metric].append(data[metric])
        
        # Calculate average similarities
        for metric in analysis['similarity_metrics']:
            if analysis['similarity_metrics'][metric]:
                analysis['similarity_metrics'][metric] = {
                    'mean': np.mean(analysis['similarity_metrics'][metric]),
                    'std': np.std(analysis['similarity_metrics'][metric])
                }
        
        # Find central materials using betweenness centrality
        centrality = nx.betweenness_centrality(G)
        analysis['central_materials'] = sorted(
            [(node, cent) for node, cent in centrality.items() if G.nodes[node]['type'] == 'material'],
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        # Detect communities
        communities = nx.community.greedy_modularity_communities(G)
        analysis['community_structure'] = [
            [node for node in comm if G.nodes[node]['type'] == 'material']
            for comm in communities
        ]
        
        return analysis

    def compute_material_properties(xyz_data, material_info):
        """Compute additional material properties from XYZ data and material info."""
        coords = np.array([coord for _, coord in xyz_data])
        properties = {}
        
        # Basic geometric properties
        center = np.mean(coords, axis=0)
        distances = np.linalg.norm(coords - center, axis=1)
        properties['center_of_mass'] = center.tolist()
        properties['max_distance_from_center'] = np.max(distances)
        properties['min_distance_from_center'] = np.min(distances)
        properties['mean_distance_from_center'] = np.mean(distances)
        
        # Volume and density calculations
        bounding_box = np.max(coords, axis=0) - np.min(coords, axis=0)
        volume = np.prod(bounding_box)
        properties['volume'] = volume
        properties['density'] = len(coords) / volume if volume > 0 else 0
        
        # Symmetry analysis with safe division
        mean_dist = np.mean(distances)
        if mean_dist > 1e-10:  # Avoid division by very small numbers
            properties['radial_symmetry'] = np.std(distances) / mean_dist
        else:
            properties['radial_symmetry'] = 0.0
        
        # Add material-specific properties
        properties.update({
            'atomic_number': material_info['atomic_number'],
            'crystal_system': material_info['crystal_system'],
            'coordination_number': material_info['coordination_number'],
            'material_type': material_info['material_type'],
            'optimization_status': material_info['optimization_status']
        })
        
        return properties

    def safe_division(a, b, epsilon=1e-10):
        """Safely compute a/b with protection against division by zero and invalid values."""
        # Convert pandas Series to numpy arrays if needed
        if hasattr(a, 'values'):
            a = a.values
        if hasattr(b, 'values'):
            b = b.values
        
        if isinstance(a, (list, tuple, np.ndarray)):
            a = np.array(a)
            b = np.array(b)
            mask = np.abs(b) > epsilon
            result = np.zeros_like(a, dtype=float)
            result[mask] = a[mask] / b[mask]
            return result
        else:
            return a / b if abs(b) > epsilon else 0.0

    def compute_similarity_metrics(props1, props2):
        """Compute similarity metrics between two materials with safe handling of edge cases."""
        metrics = {}
        
        # Radius similarity
        radius1 = props1['max_distance_from_center']
        radius2 = props2['max_distance_from_center']
        radius_diff = abs(radius1 - radius2)
        metrics['radius_similarity'] = 1.0 - (radius_diff / max(radius1, radius2)) if max(radius1, radius2) > 0 else 1.0
        
        # Structural similarity
        structural_diffs = []
        
        # Distance-based similarities
        for prop in ['max_distance_from_center', 'mean_distance_from_center']:
            val1 = props1[prop]
            val2 = props2[prop]
            if val1 > 0 or val2 > 0:  # Only compute if at least one value is non-zero
                diff = abs(val1 - val2) / (val1 + val2 + 1e-10)
                structural_diffs.append(diff)
        
        # Density similarity
        density1 = props1['density']
        density2 = props2['density']
        if density1 > 0 or density2 > 0:
            density_diff = abs(density1 - density2) / (density1 + density2 + 1e-10)
            structural_diffs.append(density_diff)
        
        metrics['structural_similarity'] = 1.0 - np.mean(structural_diffs) if structural_diffs else 1.0
        
        # Symmetry similarity
        sym1 = props1['radial_symmetry']
        sym2 = props2['radial_symmetry']
        metrics['symmetry_similarity'] = 1.0 - min(abs(sym1 - sym2), 1.0)  # Cap at 1.0
        
        return metrics

    # -------------------------------
    # 1. Build the Knowledge Graph
    # -------------------------------
    energies_df = pd.read_csv('data/energies.txt', sep=r'\s+')
    formation_df = pd.read_csv('data/formation_energies.txt', sep=r'\s+')
    material_mapping = {
        'Cu_R7_optimized': {
            'energies_key': 'R0.7', 
            'formation_key': 'R7', 
            'radius': 0.7,
            'description': 'Copper cluster with 0.7Å radius',
            'material_type': 'copper_cluster',
            'optimization_status': 'optimized',
            'atomic_number': 29,
            'crystal_system': 'cubic',
            'coordination_number': 12
        },
        'Cu_R8_optimized': {
            'energies_key': 'R0.8', 
            'formation_key': 'R8', 
            'radius': 0.8,
            'description': 'Copper cluster with 0.8Å radius',
            'material_type': 'copper_cluster',
            'optimization_status': 'optimized',
            'atomic_number': 29,
            'crystal_system': 'cubic',
            'coordination_number': 12
        },
        'Cu_R9_optimized': {
            'energies_key': 'R0.9', 
            'formation_key': 'R9', 
            'radius': 0.9,
            'description': 'Copper cluster with 0.9Å radius',
            'material_type': 'copper_cluster',
            'optimization_status': 'optimized',
            'atomic_number': 29,
            'crystal_system': 'cubic',
            'coordination_number': 12
        },
        'Cu_R10_optimized': {
            'energies_key': 'R1.0', 
            'formation_key': 'R10', 
            'radius': 1.0,
            'description': 'Copper cluster with 1.0Å radius',
            'material_type': 'copper_cluster',
            'optimization_status': 'optimized',
            'atomic_number': 29,
            'crystal_system': 'cubic',
            'coordination_number': 12
        },
    }
    
    G = nx.Graph()
    
    # Add material nodes with their properties
    for material, keys in material_mapping.items():
        # Get energy features for this specific material
        energies_features = energies_df.loc[energies_df.index == keys['energies_key']].iloc[0].to_dict()
        formation_features = formation_df.loc[formation_df.index == keys['formation_key']].iloc[0].to_dict()
        
        # Load XYZ data for the original configuration
        xyz_file = f"data/multimodal_data/{material}/xyzs/original.xyz"
        xyz_data = load_xyz_file(xyz_file)
        
        # Compute additional material properties
        material_properties = compute_material_properties(xyz_data, keys)
        
        # Combine all features
        material_features = {
            'radius': keys['radius'],
            'description': keys['description'],
            **energies_features,
            **{f'formation_{k}': v for k, v in formation_features.items()},
            **material_properties
        }
        
        # Add node for the material with enhanced features
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
            
            # Calculate energy similarity with safe division
            energy_diff = np.abs(energy1.values - energy2.values)  # Convert to numpy array
            energy_sum = np.abs(energy1.values) + np.abs(energy2.values)  # Convert to numpy array
            energy_similarity = 1.0 - np.mean(safe_division(energy_diff, energy_sum))
            
            # Get structural features for both materials
            xyz1 = load_xyz_file(f"data/multimodal_data/{mat1}/xyzs/original.xyz")
            xyz2 = load_xyz_file(f"data/multimodal_data/{mat2}/xyzs/original.xyz")
            
            # Compute structural similarities
            props1 = compute_material_properties(xyz1, material_mapping[mat1])
            props2 = compute_material_properties(xyz2, material_mapping[mat2])
            
            # Get all similarity metrics
            similarity_metrics = compute_similarity_metrics(props1, props2)
            
            # Combine similarities with weights
            if radius_diff <= 0.3:  # Only connect relatively similar materials
                radius_similarity = 1.0 - (radius_diff / 0.3)
                
                # Weighted combination of all similarity metrics
                total_similarity = (
                    0.25 * radius_similarity +      # Geometric similarity
                    0.25 * energy_similarity +      # Energy similarity
                    0.25 * similarity_metrics['structural_similarity'] +  # Structural similarity
                    0.25 * similarity_metrics['symmetry_similarity']      # Symmetry similarity
                )
                
                # Add edge with detailed similarity metrics
                G.add_edge(mat1, mat2, 
                          weight=total_similarity,
                          radius_similarity=radius_similarity,
                          energy_similarity=energy_similarity,
                          structural_similarity=similarity_metrics['structural_similarity'],
                          symmetry_similarity=similarity_metrics['symmetry_similarity'],
                          relationship_type='similar_material',
                          relationship_description=f'Materials with similar properties (radius diff: {radius_diff:.2f}Å)'
                )
    
    print("Knowledge graph built:")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    
    # Analyze the knowledge graph
    print("\nAnalyzing Knowledge Graph Structure:")
    analysis = analyze_knowledge_graph(G)
    print("\nBasic Statistics:")
    print(f"Number of nodes: {analysis['basic_stats']['num_nodes']}")
    print(f"Number of edges: {analysis['basic_stats']['num_edges']}")
    print(f"Graph density: {analysis['basic_stats']['density']:.4f}")
    print(f"Average clustering coefficient: {analysis['basic_stats']['avg_clustering']:.4f}")
    print(f"Average degree: {analysis['basic_stats']['avg_degree']:.2f}")

    print("\nSimilarity Metrics:")
    for metric, stats in analysis['similarity_metrics'].items():
        print(f"{metric}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")

    print("\nMost Central Materials:")
    for material, centrality in analysis['central_materials']:
        print(f"{material}: centrality={centrality:.4f}")

    print("\nCommunity Structure:")
    for i, community in enumerate(analysis['community_structure']):
        print(f"Community {i+1}: {', '.join(community)}")

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
            
            # Get a consistent ordering for feature keys
            self.feature_keys = sorted(list(graph.nodes[material_list[0]]['features'].keys()))
            
            # Separate numeric and non-numeric features
            self.numeric_features = []
            self.categorical_features = []
            self.feature_min = {}
            self.feature_max = {}
            self.feature_mapping = {}
            
            # First pass: identify feature types and check for constant features
            sample_features = graph.nodes[material_list[0]]['features']
            for k in self.feature_keys:
                value = sample_features[k]
                if isinstance(value, (int, float, np.number)):
                    self.numeric_features.append(k)
                else:
                    self.categorical_features.append(k)
            
            # Process numeric features
            for k in self.numeric_features:
                all_values = []
                for material in material_list:
                    feats = graph.nodes[material]['features']
                    all_values.append(feats[k])
                
                min_val = np.min(all_values)
                max_val = np.max(all_values)
                
                # Skip features that are constant across all samples
                if np.isclose(min_val, max_val):
                    print(f"Warning: Feature '{k}' has identical values across all samples: {min_val}")
                    continue
                
                self.feature_min[k] = min_val
                self.feature_max[k] = max_val
            
            # Process categorical features
            for k in self.categorical_features:
                unique_values = set()
                for material in material_list:
                    feats = graph.nodes[material]['features']
                    unique_values.add(str(feats[k]))
                
                # Skip features with only one unique value
                if len(unique_values) == 1:
                    print(f"Warning: Categorical feature '{k}' has only one unique value: {list(unique_values)[0]}")
                    continue
                
                # Create mapping for categorical features
                self.feature_mapping[k] = {val: idx for idx, val in enumerate(sorted(unique_values))}
            
            # Add rotation-specific features
            self.numeric_features.extend(['rotation_angle', 'is_original'])
            
            # Check and store label range
            all_labels = list(label_mapping.values())
            self.label_min = np.min(all_labels)
            self.label_max = np.max(all_labels)
            if np.isclose(self.label_min, self.label_max):
                raise ValueError(f"All labels have identical value: {self.label_min}. This indicates a problem with the dataset.")
            
            print(f"\nUsing {len(self.numeric_features)} numeric features and {len(self.categorical_features)} categorical features")
            print("Numeric features:", self.numeric_features)
            print("Categorical features:", self.categorical_features)
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            material, rotation = self.samples[idx]
            features, xyz_data, image_file = get_sample_data(self.graph, material, rotation)
            
            # Process features
            normalized_features = []
            for k in self.numeric_features:
                if k in ['rotation_angle', 'is_original']:
                    # Handle rotation-specific features
                    if k == 'rotation_angle':
                        normalized_features.append(rotation / 360.0)  # Normalize to [0,1]
                    else:
                        normalized_features.append(1.0 if rotation == 0 else 0.0)
                elif k in self.feature_min:  # Only process non-constant numeric features
                    # Normalize numeric features
                    min_val = self.feature_min[k]
                    max_val = self.feature_max[k]
                    norm = (features[k] - min_val) / (max_val - min_val + 1e-8)
                    normalized_features.append(norm)
            
            for k in self.categorical_features:
                if k in self.feature_mapping:  # Only process non-constant categorical features
                    # One-hot encode categorical features
                    value = str(features[k])
                    encoding = np.zeros(len(self.feature_mapping[k]))
                    encoding[self.feature_mapping[k][value]] = 1
                    normalized_features.extend(encoding)
            
            features_tensor = torch.tensor(normalized_features, dtype=torch.float)
            image = Image.open(image_file).convert('RGB')
            if self.image_transform:
                image = self.image_transform(image)
            
            formation_key = self.material_mapping[material]['formation_key']
            label_value = self.label_mapping[formation_key]
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
