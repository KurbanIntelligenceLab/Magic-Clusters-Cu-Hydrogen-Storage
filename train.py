import torch
from torch_geometric.data import Batch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Dataset, DataLoader
from model import MultiModalModel
from data_utils import load_xyz_as_pyg_data, load_image, build_tabular_tensor, load_knowledge_graph
import os
import json
from sklearn.preprocessing import StandardScaler
import numpy as np
import random
import time
from datetime import datetime


# ------------------- CONFIG -------------------
CONFIG = {
    'seeds': [42, 123], 
    'batch_size': 2,  
    'epochs': 500,
    'lr': 1e-3,  
    'weight_decay': 1e-5,  # Reduced from 1e-4
    'scheduler': {
        'T_0': 100, 
        'T_mult': 2,
        'eta_min': 1e-6
    },
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'leave_out_ids': ['R7', 'R8', 'R9', 'R10', 'R7-H2', 'R8-H2', 'R9-H2', 'R10-H2'],
    'tabular_keys': ['Cu', 'Ef_f', 'Ef_t', 'HOMO', 'LUMO', 'Eg', 'H2', "Cu-H2"],
    'target_keys': ['HOMO', 'LUMO', 'Eg', 'Ef_t', 'Ef_f'],
    'model_params': {
        'tabular_dim': 8,
        'gnn_hidden': 64,  
        'gnn_out': 32,     
        'schnet_out': 32,  
        'resnet_out': 512, #256 for resnet18
        'fusion_dim': 32,  
        'num_targets': 1
    },
    # Regularization parameters
    'dropout_rate': 0.5,  # Reduced from 0.2
    'early_stopping_patience': 300,  # Reduced from 150
    'min_delta': 5e-4,    # Reduced from 1e-3 for less sensitive early stopping
    'gradient_clip': 1.0,  # Reduced from 5 for less aggressive clipping
    # Modality tracking
    'track_modalities': True,  # Enable modality contribution tracking
    'modality_names': ['SchNet', 'Image', 'Tabular']  # Names for the three modalities
}

# ------------------- SEED SETTING -------------------
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ------------------- DATASET -------------------
class MultiModalDataset(Dataset):
    def __init__(self, nodes, tabular_keys, target_key, mask_tabular=False, tabular_scaler=None, target_scaler=None):
        self.samples = []
        self.tabular_keys = tabular_keys
        self.target_key = target_key
        self.mask_tabular = mask_tabular
        self.tabular_scaler = tabular_scaler
        self.target_scaler = target_scaler
        for node in nodes:
            for rot in node['rotations']:
                self.samples.append({
                    'node': node,
                    'rot': rot
                })
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        node = self.samples[idx]['node']
        rot = self.samples[idx]['rot']
        schnet_data = load_xyz_as_pyg_data(rot['xyz_path'])
        image = load_image(rot['image_path'])
        if self.mask_tabular:
            tabular = torch.zeros(len(self.tabular_keys), dtype=torch.float)
        else:
            tabular = build_tabular_tensor(node, self.tabular_keys)
            if self.tabular_scaler is not None:
                # Scale each feature individually using its own scaler
                scaled_features = np.zeros(len(self.tabular_keys))
                for i, key in enumerate(self.tabular_keys):
                    feature_value = tabular[i].numpy().reshape(1, -1)
                    scaled_feature = self.tabular_scaler[key].transform(feature_value)[0, 0]  # Extract scalar value
                    scaled_features[i] = scaled_feature
                tabular = torch.tensor(scaled_features, dtype=torch.float)
        target = torch.tensor([node.get(self.target_key, 0.0)], dtype=torch.float)
        if self.target_scaler is not None:
            target = torch.tensor(self.target_scaler.transform([[target.item()]])[0, 0], dtype=torch.float)
            target = target.reshape(1)  # Ensure target is 1D
        return schnet_data, image, tabular, target

def collate_fn(batch):
    schnet_data_list, image_list, tabular_list, target_list = zip(*batch)
    schnet_batch = Batch.from_data_list(schnet_data_list)
    image_batch = torch.stack(image_list)
    tabular_x = torch.stack(tabular_list)
    targets = torch.stack(target_list)
    return schnet_batch, image_batch, tabular_x, targets

# ------------------- TRAINING LOOP -------------------
def train_and_eval(kg, config):
    all_results = {}
    all_ids = [node['id'] for node in kg if node['id'] != 'H2']
    
    for seed in config['seeds']:
        print(f"\n=== Training with seed {seed} ===")
        set_seed(seed)
        results = {}
        
        for test_id in config['leave_out_ids']:
            results[test_id] = {}
            for target_key in config['target_keys']:
                print(f'Leave out: {test_id}, Target: {target_key}')
                start_time = time.time()
                
                train_nodes = [node for node in kg if node['id'] in all_ids and node['id'] != test_id]
                test_node = [node for node in kg if node['id'] == test_id][0]
                
                # --- Normalization ---
                # First, collect all values for consistent scaling
                all_values = {}
                for key in config['tabular_keys'] + [target_key]:
                    values = np.array([[node.get(key, 0.0)] for node in train_nodes])
                    all_values[key] = values
                
                # Create scalers for each feature and target
                scalers = {}
                for key, values in all_values.items():
                    scaler = StandardScaler()
                    scaler.fit(values)
                    scalers[key] = scaler
                
                # Save scalers with seed in path
                result_dir = os.path.join('results', f'seed_{seed}', test_id, target_key)
                os.makedirs(result_dir, exist_ok=True)
                
                # Save normalization statistics
                norm_stats = {
                    'scalers': {
                        key: {
                            'mean': scaler.mean_.tolist(),
                            'std': scaler.scale_.tolist(),
                            'range': {
                                'min': float(all_values[key].min()),
                                'max': float(all_values[key].max()),
                                'mean': float(scaler.mean_[0]),
                                'std': float(scaler.scale_[0])
                            },
                            'normalized_range': {
                                'min': float(scaler.transform(all_values[key]).min()),
                                'max': float(scaler.transform(all_values[key]).max()),
                                'mean': float(scaler.transform(all_values[key]).mean()),
                                'std': float(scaler.transform(all_values[key]).std())
                            }
                        }
                        for key, scaler in scalers.items()
                    }
                }
                
                with open(os.path.join(result_dir, 'normalization_stats.json'), 'w') as f:
                    json.dump(norm_stats, f, indent=2)
                
                # Save individual scalers
                for key, scaler in scalers.items():
                    with open(os.path.join(result_dir, f'{key}_scaler.json'), 'w') as f:
                        json.dump({'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist()}, f)
                
                # Build datasets with consistent scaling
                train_dataset = MultiModalDataset(
                    train_nodes, 
                    config['tabular_keys'], 
                    target_key, 
                    mask_tabular=False, 
                    tabular_scaler=scalers, 
                    target_scaler=scalers[target_key]
                )
                test_dataset = MultiModalDataset(
                    [test_node], 
                    config['tabular_keys'], 
                    target_key, 
                    mask_tabular=True, 
                    tabular_scaler=scalers, 
                    target_scaler=scalers[target_key]
                )
                train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
                test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)
                
                # Build graph: fully connected for simplicity
                N = len(train_nodes)
                edge_index = torch.combinations(torch.arange(N), r=2).t()
                edge_index = torch.cat([edge_index, edge_index[[1,0],:]], dim=1)  # undirected
                edge_index = edge_index.to(config['device'])
                
                # Model
                model = MultiModalModel(**{**config['model_params'], 'num_targets': 1, 'dropout_rate': config['dropout_rate']}).to(config['device'])
                optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
                scheduler = CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=config['scheduler']['T_0'],
                    T_mult=config['scheduler']['T_mult'],
                    eta_min=config['scheduler']['eta_min']
                )
                loss_fn = nn.MSELoss()
                model_path = os.path.join(result_dir, 'best_model.pt')
                best_test_mae = float('inf')
                best_state = None
                best_preds_inv = None
                best_trues_inv = None
                patience_counter = 0
                
                # Training
                if not os.path.exists(model_path):
                    model.train()
                    epoch_times = []
                    lr_history = []
                    modality_history = []  # Track modality contributions over epochs
                    
                    for epoch in range(config['epochs']):
                        epoch_start = time.time()
                        total_loss = 0
                        epoch_attention_weights = []  # Track attention weights for this epoch
                        
                        for schnet_batch, image_batch, tabular_x, targets in train_loader:
                            schnet_batch = schnet_batch.to(config['device'])
                            image_batch = image_batch.to(config['device'])
                            tabular_x = tabular_x.to(config['device'])
                            targets = targets.to(config['device'])
                            
                            # Build edge_index for this batch
                            batch_size = tabular_x.shape[0]
                            if batch_size > 1:
                                edge_index = torch.combinations(torch.arange(batch_size), r=2).t()
                                edge_index = torch.cat([edge_index, edge_index[[1,0],:]], dim=1)
                            else:
                                edge_index = torch.zeros((2,0), dtype=torch.long)
                            edge_index = edge_index.to(config['device'])
                            tabular_batch = torch.arange(tabular_x.shape[0], device=config['device'])
                            
                            optimizer.zero_grad()
                            fusion_out, tabular_out, image_out, schnet_out, attention_weights = model(schnet_batch, image_batch, tabular_x, edge_index, tabular_batch)
                            loss_fusion = loss_fn(fusion_out, targets)
                            loss_tabular = loss_fn(tabular_out, targets)
                            loss_image = loss_fn(image_out, targets)
                            loss_schnet = loss_fn(schnet_out, targets)
                            
                            # Modality balance regularization
                            # Encourage all modalities to contribute (prevent zero weights)
                            target_balance = torch.ones_like(attention_weights) / attention_weights.shape[1]  # Equal weights
                            balance_loss = torch.nn.functional.mse_loss(attention_weights, target_balance)
                            
                            # Entropy regularization to encourage exploration
                            entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=1).mean()
                            entropy_loss = -entropy  # Maximize entropy (minimize negative entropy)
                            
                            # Main loss + auxiliary losses + regularization
                            loss = loss_fusion #+ 0.1 * (loss_tabular+ loss_image + loss_schnet) + 0.1 * balance_loss + 0.05 * entropy_loss
                            loss.backward()
                            
                            # Gradient clipping
                            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
                            
                            optimizer.step()
                            total_loss += loss.item() * len(targets)
                            
                            # Store attention weights for this batch
                            if config['track_modalities']:
                                epoch_attention_weights.append(attention_weights.detach().cpu().numpy())
                        
                        # Calculate average attention weights for this epoch
                        if config['track_modalities'] and epoch_attention_weights:
                            avg_attention = np.mean(np.concatenate(epoch_attention_weights, axis=0), axis=0)
                            modality_history.append(avg_attention)
                        
                        # Update learning rate
                        scheduler.step()
                        current_lr = optimizer.param_groups[0]['lr']
                        lr_history.append(current_lr)
                        
                        # Evaluate on test set after each epoch
                        model.eval()
                        with torch.no_grad():
                            test_preds, test_trues = [], []
                            test_attention_weights = []
                            for schnet_batch, image_batch, tabular_x, targets in test_loader:
                                schnet_batch = schnet_batch.to(config['device'])
                                image_batch = image_batch.to(config['device'])
                                tabular_x = tabular_x.to(config['device'])
                                batch_size = tabular_x.shape[0]
                                if batch_size > 1:
                                    edge_index = torch.combinations(torch.arange(batch_size), r=2).t()
                                    edge_index = torch.cat([edge_index, edge_index[[1,0],:]], dim=1)
                                else:
                                    edge_index = torch.zeros((2,0), dtype=torch.long)
                                edge_index = edge_index.to(config['device'])
                                tabular_batch = torch.arange(tabular_x.shape[0], device=config['device'])
                                fusion_out, tabular_out, image_out, schnet_out, attention_weights = model(schnet_batch, image_batch, tabular_x, edge_index, tabular_batch)
                                test_preds.append(fusion_out.cpu())
                                test_trues.append(targets.cpu())
                                test_attention_weights.append(attention_weights.cpu().numpy())
                            
                            test_preds = torch.cat(test_preds, dim=0).numpy().flatten()
                            test_trues = torch.cat(test_trues, dim=0).numpy().flatten()
                            preds_inv = scalers[target_key].inverse_transform(test_preds.reshape(-1,1)).flatten()
                            trues_inv = scalers[target_key].inverse_transform(test_trues.reshape(-1,1)).flatten()
                            test_mae = float(np.mean(np.abs(preds_inv - trues_inv)))
                            
                            # Calculate test attention weights
                            test_avg_attention = np.mean(np.concatenate(test_attention_weights, axis=0), axis=0) if test_attention_weights else None
                            
                            # Early stopping check
                            if test_mae < best_test_mae - config['min_delta']:
                                best_test_mae = test_mae
                                best_state = model.state_dict()
                                best_preds_inv = preds_inv.copy()
                                best_trues_inv = trues_inv.copy()
                                best_attention_weights = test_avg_attention.copy() if test_avg_attention is not None else None
                                patience_counter = 0
                            else:
                                patience_counter += 1
                            
                            # Early stopping
                            if patience_counter >= config['early_stopping_patience']:
                                print(f'Early stopping at epoch {epoch} (patience: {patience_counter})')
                                break
                        
                        model.train()
                        epoch_time = time.time() - epoch_start
                        epoch_times.append(epoch_time)
                        
                        # Print modality contributions every 10 epochs
                        if epoch % 10 == 0 and config['track_modalities'] and test_avg_attention is not None:
                            modality_str = " | ".join([f"{name}: {weight:.3f}" for name, weight in zip(config['modality_names'], test_avg_attention)])
                            print(f'Epoch {epoch} Loss: {total_loss/len(train_loader.dataset):.4f} | Test MAE: {test_mae:.4f} | LR: {current_lr:.2e} | Time: {epoch_time:.2f}s | Patience: {patience_counter}')
                            print(f'  Modality Weights: {modality_str}')
                            
                            # Check for very low weights and warn
                            low_weight_modalities = [name for name, weight in zip(config['modality_names'], test_avg_attention) if weight < 0.05]
                            if low_weight_modalities:
                                print(f'  ⚠️  Low weights detected: {low_weight_modalities}')
                        elif epoch % 5 == 0:
                            print(f'Epoch {epoch} Loss: {total_loss/len(train_loader.dataset):.4f} | Test MAE: {test_mae:.4f} | LR: {current_lr:.2e} | Time: {epoch_time:.2f}s | Patience: {patience_counter}')
                    
                    # Save best model and results after training
                    torch.save(best_state, model_path)
                    total_time = time.time() - start_time
                    print(f"Saved best model to {model_path} (Test MAE: {best_test_mae:.4f})")
                    print(f"Total training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
                    
                    # Print final modality contributions
                    if config['track_modalities'] and best_attention_weights is not None:
                        print(f"\n=== Final Modality Contributions ===")
                        for name, weight in zip(config['modality_names'], best_attention_weights):
                            print(f"{name}: {weight:.4f}")
                        
                        # Check for zero or very low weights
                        zero_threshold = 0.01
                        low_weight_modalities = [name for name, weight in zip(config['modality_names'], best_attention_weights) if weight < zero_threshold]
                        if low_weight_modalities:
                            print(f"\n⚠️  WARNING: Low contribution modalities (< {zero_threshold}): {low_weight_modalities}")
                            if 'Tabular' in low_weight_modalities:
                                print("💡 Consider SHAP analysis for tabular features to understand their importance")
                    
                    results_json = {
                        'pred': best_preds_inv.tolist(),
                        'true': best_trues_inv.tolist(),
                        'mae': best_test_mae,
                        'training_time': total_time,
                        'avg_epoch_time': np.mean(epoch_times),
                        'final_epoch': epoch,
                        'early_stopped': patience_counter >= config['early_stopping_patience'],
                        'modality_contributions': {
                            'final_weights': best_attention_weights.tolist() if best_attention_weights is not None else None,
                            'modality_names': config['modality_names'],
                            'epoch_history': [weights.tolist() for weights in modality_history] if modality_history else None
                        }
                    }
                    with open(os.path.join(result_dir, 'results.json'), 'w') as f:
                        json.dump(results_json, f, indent=2)
                    results[test_id][target_key] = results_json
                else:
                    print(f"Loading model from {model_path}")
                    model.load_state_dict(torch.load(model_path, map_location=config['device']))
                    # Also load results.json if exists
                    results_json_path = os.path.join(result_dir, 'results.json')
                    if os.path.exists(results_json_path):
                        with open(results_json_path) as f:
                            results_json = json.load(f)
                        results[test_id][target_key] = results_json
                        
                        # Print modality contributions if available
                        if 'modality_contributions' in results_json and results_json['modality_contributions']['final_weights']:
                            print(f"\n=== Loaded Modality Contributions ===")
                            weights = results_json['modality_contributions']['final_weights']
                            names = results_json['modality_contributions']['modality_names']
                            for name, weight in zip(names, weights):
                                print(f"{name}: {weight:.4f}")
                            
                            # Check for low weights
                            zero_threshold = 0.01
                            low_weight_modalities = [name for name, weight in zip(names, weights) if weight < zero_threshold]
                            if low_weight_modalities:
                                print(f"\n⚠️  WARNING: Low contribution modalities (< {zero_threshold}): {low_weight_modalities}")
                                if 'Tabular' in low_weight_modalities:
                                    print("💡 Consider SHAP analysis for tabular features to understand their importance")
        
        all_results[f'seed_{seed}'] = results
    
    return all_results

if __name__ == '__main__':
    start_time = time.time()
    kg = load_knowledge_graph('new_data/knowledge_graph.json')
    results = train_and_eval(kg, CONFIG)
    total_time = time.time() - start_time
    print("\n=== Final Results ===")
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(json.dumps(results, indent=2))