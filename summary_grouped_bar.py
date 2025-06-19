import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import json
from model import MultiModalModel
from data_utils import load_knowledge_graph, build_tabular_tensor, load_xyz_as_pyg_data, load_image
import random

# --- Visualization Config (copied from summary_heatmap.py) ---
VISUALIZATION_CONFIG = {
    'dpi': 300,
    'save_format': 'pdf',
    'bbox_inches': 'tight',
    'font_family': 'serif',
    'font_size': {
        'title': 16,
        'axis_label': 14,
        'tick_label': 12,
        'legend': 12,
        'network_label': 12,
        'colorbar_label': 12,
        'annotation': 10
    },
    'font_weight': {
        'title': 'bold',
        'axis_label': 'normal',
        'tick_label': 'normal',
        'legend': 'normal',
        'network_label': 'bold',
        'colorbar_label': 'normal',
        'annotation': 'normal'
    },
    'figure_size': {
        'single': (10, 8),
        'wide': (12, 8),
        'square': (10, 10),
        'large': (15, 12),
        'extra_large': (20, 15),
        'combined': (25, 10),
        'network': (12, 12),
        'correlation': (12, 12),
        'combined_network': (33, 22),
        'combined_correlation': (20, 8)
    },
    'line_width': 2,
    'bar_width': 1,
    'bar_group_gap': 1,     # space between each tick‐group
    'alpha': 0.8,
    'edge_alpha': 0.3,
    'node_size': {
        'large': 3000,
        'medium': 2000,
        'small': 1500
    },
    'edge_width': 2,
    'cmap': 'RdYlBu_r',
    'heatmap_cmap': 'coolwarm',
    'node_color_no_pred': 'gray',
    'node_alpha_no_pred': 0.6,
    'spring_k': 2,
    'spring_iterations': 100,
    'shap_background_samples': 5,
    'shap_plot_size': (12, 12),
    'heatmap_center': 0,
    'heatmap_annot': True,
    'tight_layout': True,
    'constrained_layout': False,
    'subplots_adjust': {
        'left': 0.1,
        'right': 0.9,
        'top': 0.9,
        'bottom': 0.1,
        'wspace': 0.3,
        'hspace': 0.3
    }
}
plt.rcParams.update({
    'font.family': VISUALIZATION_CONFIG['font_family'],
    'font.size': VISUALIZATION_CONFIG['font_size']['tick_label'],
    'axes.titlesize': VISUALIZATION_CONFIG['font_size']['title'],
    'axes.labelsize': VISUALIZATION_CONFIG['font_size']['axis_label'],
    'xtick.labelsize': VISUALIZATION_CONFIG['font_size']['tick_label'],
    'ytick.labelsize': VISUALIZATION_CONFIG['font_size']['tick_label'],
    'legend.fontsize': VISUALIZATION_CONFIG['font_size']['legend'],
    'figure.titlesize': VISUALIZATION_CONFIG['font_size']['title'],
    'lines.linewidth': VISUALIZATION_CONFIG['line_width'],
    'axes.linewidth': VISUALIZATION_CONFIG['line_width'],
    'grid.linewidth': VISUALIZATION_CONFIG['line_width'] * 0.5,
})

def save_figure(filename, dpi=None, format=None, bbox_inches=None):
    dpi = dpi or VISUALIZATION_CONFIG['dpi']
    format = format or VISUALIZATION_CONFIG['save_format']
    bbox_inches = bbox_inches or VISUALIZATION_CONFIG['bbox_inches']
    plt.savefig(filename, dpi=dpi, format=format, bbox_inches=bbox_inches)
    plt.close()

# --- Real feature importance aggregation ---
def compute_real_feature_importance():
    # Config (should match your training/visualization config)
    CONFIG = {
        'seeds': [42],
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'leave_out_ids': ['R7-H2', 'R8-H2', 'R9-H2', 'R10-H2'],
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
    }
    modalities = ['Geometric', 'Image', 'Knowledge Graph']
    features = CONFIG['tabular_keys']
    methods = ['attention', 'ablation', 'gradient', 'gnn', 'perturbation', 'consensus']
    targets = CONFIG['target_keys']
    kg = load_knowledge_graph('new_data/knowledge_graph.json')
    results = {m: {t: [] for t in targets} for m in methods}
    # For each target, aggregate over all seeds and leave-out IDs
    for target in targets:
        # Containers for each method
        attn_all, ablation_all, grad_all, gnn_all, pert_all = [], [], [], [], []
        for leave_out_id in CONFIG['leave_out_ids']:
            for seed in CONFIG['seeds']:
                # Set seed for reproducibility
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
                # Load model and scalers
                result_dir = os.path.join('results', f'seed_{seed}', leave_out_id, target)
                model_path = os.path.join(result_dir, 'best_model.pt')
                target_scaler_path = os.path.join(result_dir, f'{target}_scaler.json')
                model = MultiModalModel(**CONFIG['model_params']).to(CONFIG['device'])
                model.load_state_dict(torch.load(model_path, map_location=CONFIG['device']))
                model.eval()
                with open(target_scaler_path) as f:
                    target_scaler = json.load(f)
                # Prepare test samples (all rotations)
                node_candidates = [n for n in kg if n['id'] == leave_out_id]
                if not node_candidates:
                    print(f"[DEBUG] No node found in KG for leave_out_id={leave_out_id}")
                    continue
                node = node_candidates[0]
                schnet_data_list, image_list, tabular_list = [], [], []
                for rot in node['rotations']:
                    schnet_data = load_xyz_as_pyg_data(rot['xyz_path'])
                    image = load_image(rot['image_path'])
                    tabular = build_tabular_tensor(node, CONFIG['tabular_keys'])
                    # Use the target scaler for tabular features (assumption)
                    tabular = (tabular - torch.tensor(target_scaler['mean'])) / torch.tensor(target_scaler['scale'])
                    schnet_data_list.append(schnet_data)
                    image_list.append(image)
                    tabular_list.append(tabular.unsqueeze(0))
                # For each rotation, compute all explainability methods
                for schnet_data, image, tabular_x in zip(schnet_data_list, image_list, tabular_list):
                    schnet_data = schnet_data.to(CONFIG['device'])
                    image = image.unsqueeze(0).to(CONFIG['device'])
                    tabular_x = tabular_x.to(CONFIG['device'])
                    batch = torch.zeros(tabular_x.shape[0], dtype=torch.long, device=CONFIG['device'])
                    edge_index = torch.zeros((2, 0), dtype=torch.long, device=CONFIG['device'])
                    # 1. Attention
                    # with torch.no_grad():
                    #     attn_weights = torch.softmax(model.attn, dim=0).cpu().numpy()
                    # attn_all.append(attn_weights)
                    # 2. Ablation
                    ablation = []
                    with torch.no_grad():
                        baseline_pred = model(schnet_data, image, tabular_x, edge_index, batch)
                        baseline_pred_original = target_scaler['scale'][0] * baseline_pred.cpu().numpy() + target_scaler['mean'][0]
                        for i, feature_name in enumerate(features):
                            ablated_tabular = tabular_x.clone()
                            ablated_tabular[0, i] = 0.0
                            ablated_pred = model(schnet_data, image, ablated_tabular, edge_index, batch)
                            ablated_pred_original = target_scaler['scale'][0] * ablated_pred.cpu().numpy() + target_scaler['mean'][0]
                            pred_change = abs(ablated_pred_original - baseline_pred_original)
                            pred_change_percent = (pred_change / abs(baseline_pred_original)) * 100
                            ablation.append(pred_change_percent[0].item())
                    ablation_all.append(ablation)
                    # 3. Gradient
                    tabular_x.requires_grad_(True)
                    pred = model(schnet_data, image, tabular_x, edge_index, batch)
                    pred.backward()
                    tabular_gradients = tabular_x.grad.clone().cpu().numpy().flatten()
                    tabular_x.requires_grad_(False)
                    grad_all.append(np.abs(tabular_gradients))
                    # 4. GNN (gradient-based)
                    class GNNWrapper(torch.nn.Module):
                        def __init__(self, original_model):
                            super().__init__()
                            self.original_model = original_model
                        def forward(self, x, edge_index, batch):
                            return self.original_model.tabular_gnn(x, edge_index, batch)
                    wrapper = GNNWrapper(model)
                    tabular_x.requires_grad_(True)
                    gnn_output = wrapper(tabular_x, edge_index, batch)
                    gnn_output_mean = gnn_output.mean()
                    gnn_output_mean.backward()
                    gnn_importance = torch.abs(tabular_x.grad).detach().cpu().numpy().flatten()
                    tabular_x.requires_grad_(False)
                    gnn_all.append(gnn_importance)
                    # 5. Perturbation
                    perturb_scores = []
                    with torch.no_grad():
                        baseline_pred = model(schnet_data, image, tabular_x, edge_index, batch)
                        baseline_pred_original = target_scaler['scale'][0] * baseline_pred.cpu().numpy() + target_scaler['mean'][0]
                        for i, feature_name in enumerate(features):
                            pred_changes_percent = []
                            for scale in [-0.5, -0.25, 0.25, 0.5]:
                                perturbed_tabular = tabular_x.clone()
                                perturbed_tabular[0, i] *= (1 + scale)
                                perturbed_pred = model(schnet_data, image, perturbed_tabular, edge_index, batch)
                                perturbed_pred_original = target_scaler['scale'][0] * perturbed_pred.cpu().numpy() + target_scaler['mean'][0]
                                pred_change = abs(perturbed_pred_original - baseline_pred_original)
                                pred_change_percent = (pred_change / abs(baseline_pred_original)) * 100
                                pred_changes_percent.append(pred_change_percent[0].item())
                            perturb_scores.append(np.mean(pred_changes_percent))
                    pert_all.append(perturb_scores)
        # Average over all runs (seeds x leave_out_ids x rotations)
        if attn_all:
            attn_mean = np.mean(attn_all, axis=0)
            results['attention'][target] = 100 * attn_mean / np.sum(attn_mean)  # percent, sum to 100
        if ablation_all:
            ablation_mean = np.mean(ablation_all, axis=0)
            results['ablation'][target] = ablation_mean
        if grad_all:
            grad_mean = np.mean(grad_all, axis=0)
            results['gradient'][target] = 100 * grad_mean / np.max(grad_mean)  # percent of max
        if gnn_all:
            gnn_mean = np.mean(gnn_all, axis=0)
            results['gnn'][target] = 100 * gnn_mean / np.max(gnn_mean)
        if pert_all:
            pert_mean = np.mean(pert_all, axis=0)
            results['perturbation'][target] = pert_mean  # already in % of baseline
        # Consensus: average normalized ranks of ablation, gradient, gnn, perturbation
        consensus = np.zeros(len(features))
        n_methods = 0
        for m in ['ablation', 'gradient', 'gnn', 'perturbation']:
            if m in results and target in results[m]:
                arr = results[m][target]
                if len(arr) > 0 and np.max(arr) > 0:
                    consensus += arr / np.max(arr)
                    n_methods += 1
        if n_methods > 0:
            consensus = 100 * consensus / np.max(consensus)
            results['consensus'][target] = consensus
        # --- SET ATTENTION TO EMPTY ---
        results['attention'][target] = []
    return results, targets, features, modalities, methods

def load_all_target_results():
    # Deprecated: replaced by compute_real_feature_importance()
    raise NotImplementedError('Use compute_real_feature_importance() instead.')

def plot_grouped_bar_all_targets(data, features, modalities, methods, targets, outdir):
    n_targets = len(targets)
    bar_width = 1
    group_gap = n_targets * bar_width + 1  # add more space between groups
    colors = plt.cm.tab10(np.linspace(0,1,n_targets))

    fig, axes = plt.subplots(2, 3, figsize=(3*10, 2*8))  # make the figure wider and taller
    axes = axes.flatten()
    panel_labels = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)']
    y_labels = [
        'Modality Contribution (%)',
        'Prediction Change (% of Baseline)',
        '|Gradient| (% of max)',
        'Importance (% of max)',
        'Prediction Change (% of Baseline)',
        'Consensus Score (%)'
    ]

    property_labels = {
        'Ef_f': r'$E_f$',
        'Ef_t': r'$E_t$',
        'Eg': r'$E_g$',
        'HOMO': r'$E_H$',
        'LUMO': r'$E_L$'
    }
    for m_idx, method in enumerate(methods):
        ax = axes[m_idx]
        if method == 'attention':
            x_labels = modalities
            n_x = len(x_labels)
            x_base = np.arange(n_x) * group_gap
            all_x = []
            for t_idx, target in enumerate(targets):
                vals = data[method][target]
                if len(vals) == 0:
                    continue  # Skip if no data for this target
                x = x_base + (t_idx - (n_targets - 1) / 2) * bar_width
                all_x.extend(x.tolist())
                bars = ax.bar(x, vals, width=bar_width, color=colors[t_idx], alpha=0.8, label=target)
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, height + 2, f'{height:.1f}',
                            ha='center', va='bottom', fontsize=VISUALIZATION_CONFIG['font_size']['annotation'], rotation=60, rotation_mode='anchor')
            ax.set_xticks(x_base)
            ax.set_xticklabels(x_labels, rotation=0, ha='center')
        else:
            x_labels = [property_labels.get(f, f) for f in features]
            n_x = len(x_labels)
            x_base = np.arange(n_x) * group_gap
            all_x = []
            for t_idx, target in enumerate(targets):
                vals = data[method][target]
                if len(vals) == 0:
                    continue  # Skip if no data for this target
                x = x_base + (t_idx - (n_targets - 1) / 2) * bar_width
                all_x.extend(x.tolist())
                bars = ax.bar(x, vals, width=bar_width, color=colors[t_idx], alpha=0.8, label=target)
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, height + 2, f'{height:.1f}',
                            ha='center', va='bottom', fontsize=VISUALIZATION_CONFIG['font_size']['annotation'], rotation=45, rotation_mode='anchor')
            ax.set_xticks(x_base)
            ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.set_ylabel(y_labels[m_idx], fontsize=VISUALIZATION_CONFIG['font_size']['axis_label'])
        ax.set_title(f'{panel_labels[m_idx]}', fontsize=VISUALIZATION_CONFIG['font_size']['title'], fontweight=VISUALIZATION_CONFIG['font_weight']['title'])
        ax.set_xlabel('')
        ax.set_ylim(0, 110)
        if all_x:
            x_min = min(all_x) - bar_width
            x_max = max(all_x) + bar_width
            ax.set_xlim(x_min, x_max)

    # shared legend
    handles = [plt.Rectangle((0,0),1,1,color=colors[i],label=targets[i])
               for i in range(n_targets)]
    fig.legend(handles=handles, title='Target',
               bbox_to_anchor=(1.01,1), loc='upper left')
    plt.tight_layout(rect=[0,0,0.95,1])
    save_figure(os.path.join(outdir, 'grouped_bar_all_targets.pdf'))

if __name__ == '__main__':
    EXPLAIN_DIR = 'explainability'
    os.makedirs(EXPLAIN_DIR, exist_ok=True)
    results, targets, features, modalities, methods = compute_real_feature_importance()
    plot_grouped_bar_all_targets(results, features, modalities, methods, targets, EXPLAIN_DIR) 