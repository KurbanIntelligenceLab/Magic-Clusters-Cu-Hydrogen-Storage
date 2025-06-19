import torch
import torch.nn as nn
from model import MultiModalModel
from torch_geometric.data import Data
import numpy as np

def test_attention_mechanism():
    """
    Test the attention mechanism to ensure it's not producing zero weights.
    """
    print("=== Testing Attention Mechanism ===\n")
    
    # Model parameters
    model_params = {
        'tabular_dim': 8,
        'gnn_hidden': 64,
        'gnn_out': 32,
        'schnet_out': 32,
        'resnet_out': 512,
        'fusion_dim': 32,
        'num_targets': 1,
        'dropout_rate': 0.3
    }
    
    # Create model
    model = MultiModalModel(**model_params)
    model.eval()
    
    # Create dummy data
    batch_size = 4
    
    # SchNet data (simulate 4 molecules, 5 atoms each)
    num_atoms_per_sample = 5
    num_atoms = num_atoms_per_sample * batch_size
    z = torch.randint(1, 10, (num_atoms,)).long()  # [num_atoms]
    pos = torch.randn(num_atoms, 3)
    batch_vec = torch.arange(batch_size).repeat_interleave(num_atoms_per_sample)
    # Fully connected edge index for each molecule
    edge_indices = []
    for i in range(batch_size):
        idx_start = i * num_atoms_per_sample
        idx_end = (i + 1) * num_atoms_per_sample
        nodes = torch.arange(idx_start, idx_end)
        edges = torch.combinations(nodes, r=2).t()
        edges = torch.cat([edges, edges[[1,0],:]], dim=1)  # undirected
        edge_indices.append(edges)
    edge_index = torch.cat(edge_indices, dim=1)
    schnet_data = Data(z=z, pos=pos, batch=batch_vec, edge_index=edge_index)
    
    # Image data
    image_batch = torch.randn(batch_size, 3, 224, 224)
    
    # Tabular data
    tabular_x = torch.randn(batch_size, 8)
    
    # Tabular GNN edge index (fully connected for 4 nodes)
    tabular_edge_index = torch.combinations(torch.arange(batch_size), r=2).t()
    tabular_edge_index = torch.cat([tabular_edge_index, tabular_edge_index[[1,0],:]], dim=1)
    tabular_batch = torch.arange(batch_size)
    
    print("Input shapes:")
    print(f"  SchNet data: {schnet_data.z.shape}, {schnet_data.pos.shape}")
    print(f"  Image batch: {image_batch.shape}")
    print(f"  Tabular data: {tabular_x.shape}")
    print(f"  Edge index: {edge_index.shape}")
    print(f"  Tabular batch: {tabular_batch.shape}")
    print()
    
    # Forward pass
    with torch.no_grad():
        fusion_out, tabular_out, image_out, schnet_out, attention_weights = model(
            schnet_data, image_batch, tabular_x, tabular_edge_index, tabular_batch
        )
    
    print("Output shapes:")
    print(f"  Fusion output: {fusion_out.shape}")
    print(f"  Tabular output: {tabular_out.shape}")
    print(f"  Image output: {image_out.shape}")
    print(f"  SchNet output: {schnet_out.shape}")
    print(f"  Attention weights: {attention_weights.shape}")
    print()
    
    print("Attention weights analysis:")
    print(f"  Raw attention weights:\n{attention_weights}")
    print(f"  Sum of weights per sample: {attention_weights.sum(dim=1)}")
    print(f"  Mean weights per modality: {attention_weights.mean(dim=0)}")
    print(f"  Std weights per modality: {attention_weights.std(dim=0)}")
    print(f"  Min weights per modality: {attention_weights.min(dim=0)}")
    print(f"  Max weights per modality: {attention_weights.max(dim=0)}")
    print()
    
    # Tabular GNN analysis
    print("Tabular GNN analysis:")
    print(f"  Tabular GNN output shape: {tabular_out.shape}")
    print(f"  Tabular GNN output:\n{tabular_out}")
    print(f"  Tabular GNN mean: {tabular_out.mean():.6f}")
    print(f"  Tabular GNN std: {tabular_out.std():.6f}")
    print(f"  Tabular GNN min: {tabular_out.min():.6f}")
    print(f"  Tabular GNN max: {tabular_out.max():.6f}")
    print()
    
    # Check if tabular GNN is producing different outputs
    if tabular_out.shape[0] > 1:
        tabular_diff = torch.abs(tabular_out[0] - tabular_out[1])
        print(f"  Difference between first two tabular outputs: {tabular_diff}")
        print(f"  Max difference: {tabular_diff.max():.6f}")
        print(f"  Mean difference: {tabular_diff.mean():.6f}")
        print()
    
    # Check for zero weights
    zero_weights = (attention_weights < 1e-6).any(dim=0)
    print("Zero weight check:")
    modalities = ['SchNet', 'Image', 'Tabular']
    for i, modality in enumerate(modalities):
        has_zero = zero_weights[i]
        min_weight = attention_weights[:, i].min().item()
        print(f"  {modality}: {'❌ ZERO WEIGHTS DETECTED' if has_zero else '✅ OK'} (min: {min_weight:.6f})")
    
    # Check minimum weight constraint
    min_constraint = 0.1
    below_constraint = (attention_weights < min_constraint).any(dim=0)
    print(f"\nMinimum weight constraint check (< {min_constraint}):")
    for i, modality in enumerate(modalities):
        below = below_constraint[i]
        min_weight = attention_weights[:, i].min().item()
        print(f"  {modality}: {'⚠️  Below constraint' if below else '✅ Above constraint'} (min: {min_weight:.6f})")
    
    # Tabular GNN weight analysis
    print(f"\nTabular GNN Weight Analysis:")
    print(f"  Are all tabular outputs identical? {torch.allclose(tabular_out[0], tabular_out[1]) if tabular_out.shape[0] > 1 else 'N/A (single sample)'}")
    print(f"  Tabular output variance: {tabular_out.var():.6f}")
    print(f"  Tabular output range: {tabular_out.max() - tabular_out.min():.6f}")
    
    # Check if tabular GNN is learning anything
    if tabular_out.std() < 1e-6:
        print(f"  ⚠️  WARNING: Tabular GNN outputs have very low variance (std={tabular_out.std():.6f})")
        print(f"     This suggests the GNN might not be learning or the inputs are too similar")
    else:
        print(f"  ✅ Tabular GNN is producing varied outputs (std={tabular_out.std():.6f})")
    
    print(f"\n✅ Attention mechanism test completed!")
    return attention_weights

if __name__ == '__main__':
    attention_weights = test_attention_mechanism() 