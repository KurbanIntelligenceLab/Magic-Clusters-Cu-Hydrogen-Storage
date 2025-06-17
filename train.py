import torch
from torch_geometric.data import Batch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import MultiModalModel
from data_utils import load_xyz_as_pyg_data, load_image, build_tabular_tensor, load_knowledge_graph
import os
import json
from sklearn.preprocessing import StandardScaler
import numpy as np


# ------------------- CONFIG -------------------
CONFIG = {
    'batch_size': 6,
    'epochs': 100,
    'lr': 1e-3,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'leave_out_ids': ['R7-H2', 'R8-H2', 'R9-H2', 'R10-H2'],
    'tabular_keys': ['Cu', 'Ef_f', 'Ef_t', 'HOMO', 'LUMO', 'Eg'],
    'target_keys': ['HOMO', 'LUMO', 'Eg', 'Ef_t', 'Ef_f'],
    'model_params': {
        'tabular_dim': 6,
        'gnn_hidden': 32,
        'gnn_out': 32,
        'schnet_out': 64,
        'resnet_out': 2048,
        'fusion_dim': 128,
        'num_targets': 1
    }
}

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
                tabular = torch.tensor(self.tabular_scaler.transform([tabular.numpy()])[0], dtype=torch.float)
        target = torch.tensor([node.get(self.target_key, 0.0)], dtype=torch.float)
        if self.target_scaler is not None:
            target = torch.tensor(self.target_scaler.transform([[target.item()]])[0], dtype=torch.float)
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
    results = {}
    all_ids = [node['id'] for node in kg if node['id'] != 'H2']
    for test_id in config['leave_out_ids']:
        results[test_id] = {}
        for target_key in config['target_keys']:
            print(f'Leave out: {test_id}, Target: {target_key}')
            train_nodes = [node for node in kg if node['id'] in all_ids and node['id'] != test_id]
            test_node = [node for node in kg if node['id'] == test_id][0]
            # --- Normalization ---
            # Fit scalers on training set
            tabular_scaler = StandardScaler()
            target_scaler = StandardScaler()
            tabular_mat = np.array([build_tabular_tensor(node, config['tabular_keys']).numpy() for node in train_nodes])
            target_vec = np.array([[node.get(target_key, 0.0)] for node in train_nodes])
            tabular_scaler.fit(tabular_mat)
            target_scaler.fit(target_vec)
            # Save scalers
            result_dir = os.path.join('results', test_id, target_key)
            os.makedirs(result_dir, exist_ok=True)
            with open(os.path.join(result_dir, 'tabular_scaler.json'), 'w') as f:
                json.dump({'mean': tabular_scaler.mean_.tolist(), 'scale': tabular_scaler.scale_.tolist()}, f)
            with open(os.path.join(result_dir, 'target_scaler.json'), 'w') as f:
                json.dump({'mean': target_scaler.mean_.tolist(), 'scale': target_scaler.scale_.tolist()}, f)
            # Build datasets
            train_dataset = MultiModalDataset(train_nodes, config['tabular_keys'], target_key, mask_tabular=False, tabular_scaler=tabular_scaler, target_scaler=target_scaler)
            test_dataset = MultiModalDataset([test_node], config['tabular_keys'], target_key, mask_tabular=True, tabular_scaler=tabular_scaler, target_scaler=target_scaler)
            train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
            test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)
            # Build graph: fully connected for simplicity
            N = len(train_nodes)
            edge_index = torch.combinations(torch.arange(N), r=2).t()
            edge_index = torch.cat([edge_index, edge_index[[1,0],:]], dim=1)  # undirected
            edge_index = edge_index.to(config['device'])
            # Model
            model = MultiModalModel(**{**config['model_params'], 'num_targets': 1}).to(config['device'])
            optimizer = optim.Adam(model.parameters(), lr=config['lr'])
            loss_fn = nn.MSELoss()
            model_path = os.path.join(result_dir, 'best_model.pt')
            best_mae = float('inf')
            best_state = None
            # Training
            if not os.path.exists(model_path):
                model.train()
                for epoch in range(config['epochs']):
                    total_loss = 0
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
                        out = model(schnet_batch, image_batch, tabular_x, edge_index, tabular_batch)
                        loss = loss_fn(out, targets)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item() * len(targets)
                    # Validation on train set for best model selection (since no val set, use train MAE)
                    model.eval()
                    with torch.no_grad():
                        all_preds, all_trues = [], []
                        for schnet_batch, image_batch, tabular_x, targets in train_loader:
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
                            preds = model(schnet_batch, image_batch, tabular_x, edge_index, tabular_batch)
                            all_preds.append(preds.cpu())
                            all_trues.append(targets.cpu())
                        all_preds = torch.cat(all_preds, dim=0)
                        all_trues = torch.cat(all_trues, dim=0)
                        mae = torch.mean(torch.abs(all_preds - all_trues)).item()
                        if mae < best_mae:
                            best_mae = mae
                            best_state = model.state_dict()
                    model.train()
                    if epoch % 20 == 0:
                        print(f'Epoch {epoch} Loss: {total_loss/len(train_loader.dataset):.4f} | Train MAE: {mae:.4f}')
                # Save best model after training
                torch.save(best_state, model_path)
                print(f"Saved best model to {model_path} (MAE: {best_mae:.4f})")
            else:
                print(f"Loading model from {model_path}")
                model.load_state_dict(torch.load(model_path, map_location=config['device']))
            # Testing
            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for schnet_batch, image_batch, tabular_x, targets in test_loader:
                    schnet_batch = schnet_batch.to(config['device'])
                    image_batch = image_batch.to(config['device'])
                    tabular_x = tabular_x.to(config['device'])
                    # For test, use a single node graph (no edges)
                    batch_size = tabular_x.shape[0]
                    if batch_size > 1:
                        edge_index = torch.combinations(torch.arange(batch_size), r=2).t()
                        edge_index = torch.cat([edge_index, edge_index[[1,0],:]], dim=1)
                    else:
                        edge_index = torch.zeros((2,0), dtype=torch.long)
                    edge_index = edge_index.to(config['device'])
                    tabular_batch = torch.arange(tabular_x.shape[0], device=config['device'])
                    pred = model(
                        schnet_batch, image_batch, tabular_x,
                        edge_index,
                        tabular_batch
                    )
                    preds.append(pred.cpu())
                    trues.append(targets.cpu())
            preds = torch.cat(preds, dim=0).numpy().flatten()
            trues = torch.cat(trues, dim=0).numpy().flatten()
            # Inverse transform to original scale
            preds_inv = target_scaler.inverse_transform(preds.reshape(-1,1)).flatten()
            trues_inv = target_scaler.inverse_transform(trues.reshape(-1,1)).flatten()
            mae = float(np.mean(np.abs(preds_inv - trues_inv)))
            print(f'Predictions for {test_id} ({target_key}):', preds_inv)
            print(f'True values for {test_id} ({target_key}):', trues_inv)
            print(f'Test MAE for {test_id} ({target_key}): {mae:.4f}')
            # Save results
            results_json = {
                'pred': preds_inv.tolist(),
                'true': trues_inv.tolist(),
                'mae': mae
            }
            with open(os.path.join(result_dir, 'results.json'), 'w') as f:
                json.dump(results_json, f, indent=2)
            results[test_id][target_key] = results_json
    return results

if __name__ == '__main__':
    kg = load_knowledge_graph('data/knowledge_graph.json')
    results = train_and_eval(kg, CONFIG)
    print(results)