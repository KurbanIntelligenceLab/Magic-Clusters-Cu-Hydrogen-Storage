import os
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.preprocessing import StandardScaler
import torch
from model import MultiModalModel
from data_utils import build_tabular_tensor, load_knowledge_graph
from torch.utils.data import DataLoader
import pandas as pd

# Standardized visualization configuration
VISUALIZATION_CONFIG = {
    # Figure settings
    'dpi': 300,
    'save_format': 'pdf',
    'bbox_inches': 'tight',
    
    # Font settings
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
    
    # Figure sizes
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
    
    # Plot settings
    'line_width': 2,
    'bar_width': 0.8,
    'alpha': 0.8,
    'edge_alpha': 0.3,
    'node_size': {
        'large': 3000,
        'medium': 2000,
        'small': 1500
    },
    'edge_width': 2,
    
    # Color settings
    'cmap': 'RdYlBu_r',
    'heatmap_cmap': 'coolwarm',
    'node_color_no_pred': 'gray',
    'node_alpha_no_pred': 0.6,
    
    # Network layout settings
    'spring_k': 2,
    'spring_iterations': 100,
    
    # SHAP settings
    'shap_background_samples': 5,
    'shap_plot_size': (12, 12),
    
    # Heatmap settings
    'heatmap_center': 0,
    'heatmap_annot': True,
    
    # Spacing settings
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

# Apply global matplotlib settings
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
    """Standardized figure saving function."""
    dpi = dpi or VISUALIZATION_CONFIG['dpi']
    format = format or VISUALIZATION_CONFIG['save_format']
    bbox_inches = bbox_inches or VISUALIZATION_CONFIG['bbox_inches']
    
    plt.savefig(filename, dpi=dpi, format=format, bbox_inches=bbox_inches)
    plt.close()
    print(f"Saved as {filename}")

def load_knowledge_graph():
    """Load the knowledge graph from JSON file."""
    with open('new_data/knowledge_graph.json', 'r') as f:
        return json.load(f)

def load_results():
    """Load and average results from all seeds, structures, and properties."""
    results_dir = 'results'
    all_results = {}
    seed_dirs = [d for d in os.listdir(results_dir) if d.startswith('seed_')]
    if not seed_dirs:
        print("No seed directories found in results.")
        return all_results

    # Get all structures and properties from the first seed
    first_seed = seed_dirs[0]
    structure_dirs = [d for d in os.listdir(os.path.join(results_dir, first_seed)) if d.startswith('R')]
    for structure in structure_dirs:
        all_results[structure] = {}
        property_dirs = [d for d in os.listdir(os.path.join(results_dir, first_seed, structure)) if not d.startswith('.')]
        for prop in property_dirs:
            preds, trues, maes = [], [], []
            for seed in seed_dirs:
                result_file = os.path.join(results_dir, seed, structure, prop, 'results.json')
                if os.path.exists(result_file):
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                        preds.append(np.array(data['pred']))
                        trues.append(np.array(data['true']))
                        maes.append(data['mae'])
            if preds:
                # Average across seeds
                avg_pred = np.mean(preds, axis=0).tolist()
                avg_true = np.mean(trues, axis=0).tolist()
                avg_mae = float(np.mean(maes))
                all_results[structure][prop] = {
                    'pred': avg_pred,
                    'true': avg_true,
                    'mae': avg_mae
                }
    return all_results

def prepare_data_for_shap(kg, target_property, config):
    """Prepare data for SHAP analysis, using all nodes and filling missing keys with 0.0. Includes debug prints."""
    tabular_data = []
    used_ids = []
    for node in kg:
        tabular = build_tabular_tensor(node, config['tabular_keys'])
        tabular_data.append(tabular.numpy())
        used_ids.append(node.get('id', 'NO_ID'))
    if not tabular_data:
        print(f"prepare_data_for_shap: No tabular data for {target_property}")
        return None
    tabular_data = np.array(tabular_data)
    scaler = StandardScaler()
    tabular_data_scaled = scaler.fit_transform(tabular_data)
    return torch.tensor(tabular_data_scaled, dtype=torch.float)


def create_combined_visualization(kg, results, config):
    """Create a combined visualization with feature importance and SHAP plots for all targets, averaging across all seeds."""
    target_properties = ['HOMO', 'LUMO', 'Eg', 'Ef_t', 'Ef_f']
    property_labels = {
        'Ef_f': r'$E_f$',
        'Ef_t': r'$E_t$',
        'Eg': r'$E_g$',
        'HOMO': r'$E_H$',
        'LUMO': r'$E_L$'
    }
    feature_latex_labels = {k: property_labels.get(k, (r'$H_2$' if k == 'H2' else k)) for k in config['tabular_keys']}
    seed_dirs = [d for d in os.listdir('results') if d.startswith('seed_')]
    all_model_data = {}
    for target in target_properties:
        all_model_data[target] = []
        for leave_out_id in config['leave_out_ids']:
            for seed in seed_dirs:
                model_path = os.path.join('results', seed, leave_out_id, target, 'best_model.pt')
                if not os.path.exists(model_path):
                    print(f"Missing: {model_path}")
                else:
                    model = MultiModalModel(**config['model_params']).to(config['device'])
                    model.load_state_dict(torch.load(model_path, map_location=config['device']))
                    model.eval()
                    print("GCN conv1 weights:", model.tabular_gnn.conv1.lin.weight)
                    print("GCN conv2 weights:", model.tabular_gnn.conv2.lin.weight)
                    data = prepare_data_for_shap(kg, target, config)
                    if data is not None:
                        all_model_data[target].append((model, data))
        print(f"Loaded {len(all_model_data[target])} models for target {target}")
    fig, axes = plt.subplots(2, 5, figsize=VISUALIZATION_CONFIG['figure_size']['combined'])
    top_labels = ['(A)', '(B)', '(C)', '(D)', '(E)']
    bottom_labels = ['(F)', '(G)', '(H)', '(I)', '(J)']
    for i, target in enumerate(target_properties):
        if all_model_data[target]:
            shap_values_list = []
            for model, data in all_model_data[target]:
                try:
                    # --- Prune to just the tabular branch ---
                    tabular_gnn = model.tabular_gnn
                    tabular_gnn.eval()
                    class StandaloneTabularGNN(torch.nn.Module):
                        def __init__(self, tabular_gnn):
                            super().__init__()
                            self.tabular_gnn = tabular_gnn
                        def forward(self, x):
                            N = x.size(0)
                            edge_index = torch.arange(N, device=x.device).repeat(2, 1)  # self-loops
                            batch = torch.arange(N, device=x.device)
                            return self.tabular_gnn(x, edge_index, batch)
                    pruned_model = StandaloneTabularGNN(tabular_gnn).to(config['device'])
                    pruned_model.eval()
                    data_np = data.numpy()
                    def predict_fn(x):
                        x_tensor = torch.tensor(x, dtype=torch.float, device=config['device'])
                        with torch.no_grad():
                            out = pruned_model(x_tensor).cpu().numpy()
                            print("Model output for SHAP input:", out)
                            return out
                    background_data = data_np[:min(VISUALIZATION_CONFIG['shap_background_samples'], len(data_np))]
                    explainer = shap.KernelExplainer(predict_fn, background_data)
                    shap_values = explainer.shap_values(data_np)
                    if len(shap_values.shape) > 2:
                        shap_values = shap_values.reshape(shap_values.shape[0], -1)
                    if shap_values.shape[1] != data_np.shape[1]:
                        shap_values = shap_values[:, :data_np.shape[1]]
                    shap_values_list.append(shap_values)
                except Exception as e:
                    print(f"SHAP calculation failed for {target}: {e}")
            mean_shap = np.abs(np.mean(np.stack(shap_values_list), axis=0)).mean(axis=0)
            print(f"mean_shap for {target}: {mean_shap}")
            feature_names = [feature_latex_labels.get(key, key) for key in config['tabular_keys']]
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': mean_shap
            })
            feature_importance = feature_importance.sort_values('Importance', ascending=True)
            axes[0, i].barh(feature_importance['Feature'], feature_importance['Importance'], 
                           height=VISUALIZATION_CONFIG['bar_width'], color='#5dade2')
            axes[0, i].set_title(f'{top_labels[i]}')
            axes[0, i].set_xlabel('|SHAP|')
            axes[0, i].set_ylabel('')
        else:
            axes[0, i].text(0.5, 0.5, f'No data for {target}', ha='center', va='center', transform=axes[0, i].transAxes)
            axes[0, i].set_title(f'{top_labels[i]}')
    for i, target in enumerate(target_properties):
        if all_model_data[target]:
            shap_values_list = []
            for model, data in all_model_data[target]:
                try:
                    class TabularModelWrapper(torch.nn.Module):
                        def __init__(self, original_model):
                            super().__init__()
                            self.original_model = original_model
                        def forward(self, x):
                            N = x.size(0)
                            edge_index = torch.arange(N, device=x.device).repeat(2, 1)  # self-loops
                            batch = torch.arange(N, device=x.device)
                            return self.original_model.tabular_gnn(x, edge_index, batch)
                    wrapper_model = TabularModelWrapper(model).to(config['device'])
                    wrapper_model.eval()
                    data_np = data.numpy()
                    def predict_fn(x):
                        x_tensor = torch.tensor(x, dtype=torch.float, device=config['device'])
                        with torch.no_grad():
                            return wrapper_model(x_tensor).cpu().numpy()
                    background_data = data_np[:min(VISUALIZATION_CONFIG['shap_background_samples'], len(data_np))]
                    explainer = shap.KernelExplainer(predict_fn, background_data)
                    shap_values = explainer.shap_values(data_np)
                    if len(shap_values.shape) > 2:
                        shap_values = shap_values.reshape(shap_values.shape[0], -1)
                    if shap_values.shape[1] != data_np.shape[1]:
                        shap_values = shap_values[:, :data_np.shape[1]]
                    shap_values_list.append(shap_values)
                except Exception as e:
                    print(f"SHAP calculation failed for {target}: {e}")
            mean_shap = np.abs(np.mean(np.stack(shap_values_list), axis=0)).mean(axis=0)
            print(f"mean_shap for {target}: {mean_shap}")
            feature_names = [feature_latex_labels.get(key, key) for key in config['tabular_keys']]
            sorted_indices = np.argsort(mean_shap)
            axes[1, i].violinplot([np.concatenate([sv[:, j] for sv in shap_values_list]) for j in sorted_indices], positions=range(len(sorted_indices)))
            axes[1, i].set_xticks(range(len(sorted_indices)))
            axes[1, i].set_xticklabels([feature_names[j] for j in sorted_indices], rotation=45, ha='right')
            axes[1, i].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[1, i].set_title(f'{bottom_labels[i]}')
            if i == 0:
                axes[1, i].set_ylabel('SHAP')
            else:
                axes[1, i].set_ylabel('')
        else:
            axes[1, i].text(0.5, 0.5, f'No data for {target}', ha='center', va='center', transform=axes[1, i].transAxes)
            axes[1, i].set_title(f'{bottom_labels[i]}')
            axes[1, i].set_ylabel('')
    if VISUALIZATION_CONFIG['tight_layout']:
        plt.tight_layout()
    filename = f'combined_visualization.{VISUALIZATION_CONFIG["save_format"]}'
    save_figure(filename)

def create_combined_correlation_matrix(results):
    """Create a combined visualization with both correlation matrices side by side."""
    property_labels = {
        'Ef_f': r'$E_f$',
        'Ef_t': r'$E_t$',
        'Eg': r'$E_g$',
        'HOMO': r'$E_H$',
        'LUMO': r'$E_L$'
    }
    structure_labels = {
        'R7-H2': r'$Cu_{0.7}-{H_2}$',
        'R8-H2': r'$Cu_{0.8}-{H_2}$',
        'R9-H2': r'$Cu_{0.9}-{H_2}$',
        'R10-H2': r'$Cu_{1.0}-{H_2}$',
        'R7': r'$Cu_{0.7}$',
        'R8': r'$Cu_{0.8}$',
        'R9': r'$Cu_{0.9}$',
        'R10': r'$Cu_{1.0}$',
    }
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=VISUALIZATION_CONFIG['figure_size']['combined_correlation'])
    # Left plot: Correlation matrix of prediction errors
    data_errors = []
    for structure, properties in results.items():
        row = {'Structure': structure_labels.get(structure, structure)}
        for prop, values in properties.items():
            pred = np.array(values['pred'])
            true = np.array(values['true'])
            error = np.mean(np.abs(pred - true))
            row[f'{property_labels.get(prop, prop)}'] = error
        data_errors.append(row)
    df_errors = pd.DataFrame(data_errors)
    df_errors.set_index('Structure', inplace=True)
    corr_matrix_errors = df_errors.corr()
    sns.heatmap(corr_matrix_errors, annot=VISUALIZATION_CONFIG['heatmap_annot'], 
                cmap=VISUALIZATION_CONFIG['heatmap_cmap'], 
                center=VISUALIZATION_CONFIG['heatmap_center'], ax=ax1)
    ax1.set_title('(A)')
    # Right plot: Correlation matrix of properties
    data_properties = []
    for structure, properties in results.items():
        row = {'Structure': structure_labels.get(structure, structure)}
        for prop, values in properties.items():
            true = np.mean(np.array(values['true']))
            row[property_labels.get(prop, prop)] = true
        data_properties.append(row)
    df_properties = pd.DataFrame(data_properties)
    df_properties.set_index('Structure', inplace=True)
    corr_matrix_properties = df_properties.corr()
    sns.heatmap(corr_matrix_properties, annot=VISUALIZATION_CONFIG['heatmap_annot'], 
                cmap=VISUALIZATION_CONFIG['heatmap_cmap'], 
                center=VISUALIZATION_CONFIG['heatmap_center'], ax=ax2)
    ax2.set_title('(B)')
    if VISUALIZATION_CONFIG['tight_layout']:
        plt.tight_layout()
    filename = f'combined_correlation_matrix.{VISUALIZATION_CONFIG["save_format"]}'
    save_figure(filename)


def create_combined_network_visualization(kg, results):
    """Create a combined network visualization with all 5 target properties."""
    target_properties = ['HOMO', 'LUMO', 'Eg', 'Ef_t', 'Ef_f']
    latex_labels = {
        'R7-H2': r'$Cu_{0.7}-{H_2}$', 'R8-H2': r'$Cu_{0.8}-{H_2}$',
        'R9-H2': r'$Cu_{0.9}-{H_2}$', 'R10-H2': r'$Cu_{1.0}-{H_2}$',
        'R7':   r'$Cu_{0.7}$',           'R8':   r'$Cu_{0.8}$',
        'R9':   r'$Cu_{0.9}$',           'R10':  r'$Cu_{1.0}$',
    }
    property_labels = {
        'Ef_f': r'$E_f$', 'Ef_t': r'$E_t$',
        'Eg':   r'$E_g$', 'HOMO': r'$E_H$',
        'LUMO': r'$E_L$'
    }
    fig = plt.figure(figsize=VISUALIZATION_CONFIG['figure_size']['combined_network'], 
                     constrained_layout=VISUALIZATION_CONFIG['constrained_layout'])
    gs  = fig.add_gridspec(
        nrows=2, ncols=4,
        width_ratios =[1, 1, 1, 0.05],
        height_ratios=[1, 1]
    )
    axes_top    = [fig.add_subplot(gs[0, i]) for i in range(3)]
    axes_bottom = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 2])]
    cbar_ax     = fig.add_subplot(gs[:, 3])
    all_axes      = axes_top + axes_bottom
    last_pred_err = None
    net_labels = ['(A)', '(B)', '(C)', '(D)', '(E)']
    for ax, prop, net_label in zip(all_axes, target_properties, net_labels):
        G = nx.Graph()
        valid_ids = ['R7','R8','R9','R10','R7-H2','R8-H2','R9-H2','R10-H2']
        for rec in kg:
            nid = rec['id']
            if nid not in valid_ids:
                continue
            G.add_node(nid)
            if nid in results and prop in results[nid]:
                pred = np.array(results[nid][prop]['pred'])
                tru  = np.array(results[nid][prop]['true'])
                err  = float(np.mean(np.abs(pred - tru)))
                G.nodes[nid]['error']          = err
                G.nodes[nid]['has_prediction'] = True
            else:
                G.nodes[nid]['has_prediction'] = False
        for b in ['R7','R8','R9','R10']:
            h2 = f"{b}-H2"
            if b in G and h2 in G:
                G.add_edge(b, h2)
        ids = list(G.nodes())
        for i, n1 in enumerate(ids):
            for n2 in ids[i+1:]:
                r1 = next(r for r in kg if r['id']==n1)
                r2 = next(r for r in kg if r['id']==n2)
                if len(set(r1.keys()) & set(r2.keys())) > 1:
                    G.add_edge(n1, n2)
        pos          = nx.spring_layout(G, k=VISUALIZATION_CONFIG['spring_k'], 
                                       iterations=VISUALIZATION_CONFIG['spring_iterations'])
        sorted_nodes = sorted(
            G.nodes(),
            key=lambda x: (int(x.split('R')[1].split('-')[0]), x.endswith('-H2'))
        )
        preds = [n for n in sorted_nodes if G.nodes[n]['has_prediction']]
        errs  = [G.nodes[n]['error'] for n in preds]
        last_pred_err = errs
        if preds:
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=preds,
                node_color=errs,
                cmap=VISUALIZATION_CONFIG['cmap'],
                node_size=VISUALIZATION_CONFIG['node_size']['large'],
                alpha=VISUALIZATION_CONFIG['alpha'],
                ax=ax
            )
        nonp = [n for n in sorted_nodes if not G.nodes[n]['has_prediction']]
        if nonp:
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=nonp,
                node_color=VISUALIZATION_CONFIG['node_color_no_pred'],
                node_size=VISUALIZATION_CONFIG['node_size']['medium'],
                alpha=VISUALIZATION_CONFIG['node_alpha_no_pred'],
                ax=ax
            )
        nx.draw_networkx_edges(G, pos, alpha=VISUALIZATION_CONFIG['edge_alpha'], 
                              width=VISUALIZATION_CONFIG['edge_width'], ax=ax)
        labels = {n: latex_labels.get(n, n) for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, 
                               font_size=VISUALIZATION_CONFIG['font_size']['network_label'], 
                               font_weight=VISUALIZATION_CONFIG['font_weight']['network_label'], ax=ax)
        ax.set_title(f'{net_label}')
        ax.axis('off')
        ax.set_aspect('equal')
    if last_pred_err:
        norm = plt.Normalize(vmin=min(last_pred_err), vmax=max(last_pred_err))
        sm   = plt.cm.ScalarMappable(cmap=VISUALIZATION_CONFIG['cmap'], norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cbar_ax, orientation='vertical')
        cbar.set_label('Mean Absolute Error', 
                      fontsize=VISUALIZATION_CONFIG['font_size']['colorbar_label'])
    filename = f'combined_network_visualization.{VISUALIZATION_CONFIG["save_format"]}'
    save_figure(filename)

def main():
    # Load configuration
    CONFIG = {
        'seed': 42,
        'batch_size': 6,
        'epochs': 300,
        'lr': 1e-3,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'leave_out_ids': ['R7-H2', 'R8-H2', 'R9-H2', 'R10-H2', 'R7', 'R8', 'R9', 'R10'],
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
    
    # Load knowledge graph and results
    kg = load_knowledge_graph()
    results = load_results()
    
    # Create combined visualization
    create_combined_visualization(kg, results, CONFIG)
    
    # Create combined network visualization
    create_combined_network_visualization(kg, results)
    
    # Create combined correlation matrix
    create_combined_correlation_matrix(results)
    
if __name__ == '__main__':
    main() 