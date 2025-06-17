import torch
import os
import json
import numpy as np
from model import MultiModalModel
from data_utils import load_xyz_as_pyg_data, load_image, build_tabular_tensor, load_knowledge_graph
from torch_geometric.data import Batch
from torch_geometric.nn import GNNExplainer
from captum.attr import IntegratedGradients

# --- CONFIG ---
RESULTS_DIR = 'results'
LEAVE_OUT_ID = 'R7-H2'  # Example
TARGET_KEY = 'HOMO'     # Example
CONFIG = {
    'tabular_keys': ['Cu', 'Ef_f', 'Ef_t', 'HOMO', 'LUMO', 'Eg'],
    'model_params': {
        'tabular_dim': 6,
        'gnn_hidden': 32,
        'gnn_out': 32,
        'schnet_out': 64,
        'resnet_out': 2048,
        'fusion_dim': 128,
        'num_targets': 1
    },
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# --- Load model and scalers ---
def load_model_and_scalers(leave_out_id, target_key):
    result_dir = os.path.join(RESULTS_DIR, leave_out_id, target_key)
    model_path = os.path.join(result_dir, 'best_model.pt')
    tabular_scaler_path = os.path.join(result_dir, 'tabular_scaler.json')
    target_scaler_path = os.path.join(result_dir, 'target_scaler.json')
    model = MultiModalModel(**CONFIG['model_params']).to(CONFIG['device'])
    model.load_state_dict(torch.load(model_path, map_location=CONFIG['device']))
    with open(tabular_scaler_path) as f:
        tabular_scaler = json.load(f)
    with open(target_scaler_path) as f:
        target_scaler = json.load(f)
    return model, tabular_scaler, target_scaler

# --- Prepare a test sample ---
def prepare_test_sample(kg, leave_out_id, target_key, tabular_scaler):
    node = [n for n in kg if n['id'] == leave_out_id][0]
    rot = node['rotations'][0]  # Use first rotation for demo
    schnet_data = load_xyz_as_pyg_data(rot['xyz_path'])
    image = load_image(rot['image_path'])
    tabular = build_tabular_tensor(node, CONFIG['tabular_keys'])
    # Normalize tabular
    tabular = (tabular - torch.tensor(tabular_scaler['mean'])) / torch.tensor(tabular_scaler['scale'])
    return schnet_data, image, tabular.unsqueeze(0)

# --- GNNExplainer for Tabular GNN ---
def explain_gnn(model, tabular_x, batch):
    print("\n--- GNNExplainer (Tabular GNN) ---")
    explainer = GNNExplainer(model.tabular_gnn, epochs=200)
    # Dummy edge_index for single node
    edge_index = torch.zeros((2,0), dtype=torch.long)
    explanation = explainer.explain_node(0, tabular_x, edge_index, batch=batch)
    print('Node feature mask (importance):', explanation.node_feat_mask.detach().cpu().numpy())
    # Optionally visualize (requires matplotlib)
    # explainer.visualize_subgraph(0, edge_index, explanation.edge_mask, y=None)

# --- Captum for Tabular Input ---
def explain_tabular_captum(model, schnet_data, image, tabular_x, edge_index, batch):
    print("\n--- Captum Integrated Gradients (Tabular) ---")
    model.eval()
    def forward_tab(tab_x):
        return model(schnet_data, image, tab_x, edge_index, batch)
    ig = IntegratedGradients(forward_tab)
    attr = ig.attribute(tabular_x, target=0)
    print('Tabular feature attributions:', attr.detach().cpu().numpy())

# --- Captum for Image Input ---
def explain_image_captum(model, schnet_data, image, tabular_x, edge_index, batch):
    print("\n--- Captum Integrated Gradients (Image) ---")
    model.eval()
    def forward_img(img_x):
        return model(schnet_data, img_x, tabular_x, edge_index, batch)
    ig = IntegratedGradients(forward_img)
    attr = ig.attribute(image, target=0)
    print('Image attributions shape:', attr.shape)
    # Optionally visualize with matplotlib

if __name__ == '__main__':
    kg = load_knowledge_graph('new_data/knowledge_graph.json')
    model, tabular_scaler, target_scaler = load_model_and_scalers(LEAVE_OUT_ID, TARGET_KEY)
    schnet_data, image, tabular_x = prepare_test_sample(kg, LEAVE_OUT_ID, TARGET_KEY, tabular_scaler)
    schnet_data = schnet_data.to(CONFIG['device'])
    image = image.unsqueeze(0).to(CONFIG['device'])
    tabular_x = tabular_x.to(CONFIG['device'])
    batch = torch.zeros(tabular_x.shape[0], dtype=torch.long, device=CONFIG['device'])
    edge_index = torch.zeros((2,0), dtype=torch.long, device=CONFIG['device'])
    # GNNExplainer
    explain_gnn(model, tabular_x, batch)
    # Captum for tabular
    explain_tabular_captum(model, schnet_data, image, tabular_x, edge_index, batch)
    # Captum for image
    explain_image_captum(model, schnet_data, image, tabular_x, edge_index, batch) 