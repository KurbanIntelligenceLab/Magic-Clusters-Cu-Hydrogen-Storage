import torch
import os
import json
import numpy as np
from model import MultiModalModel
from data_utils import load_xyz_as_pyg_data, load_image, build_tabular_tensor, load_knowledge_graph
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt

# --- Visualization Config (copied from visualize_results.py) ---
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
    'bar_width': 0.8,
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
    print(f"Saved as {filename}")

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
    model.eval()
    
    with open(tabular_scaler_path) as f:
        tabular_scaler = json.load(f)
    with open(target_scaler_path) as f:
        target_scaler = json.load(f)
    return model, tabular_scaler, target_scaler

# --- Prepare all test samples (all rotations) ---
def prepare_all_test_samples(kg, leave_out_id, target_key, tabular_scaler):
    node = [n for n in kg if n['id'] == leave_out_id][0]
    schnet_data_list = []
    image_list = []
    tabular_list = []
    for rot in node['rotations']:
        schnet_data = load_xyz_as_pyg_data(rot['xyz_path'])
        image = load_image(rot['image_path'])
        tabular = build_tabular_tensor(node, CONFIG['tabular_keys'])
        tabular = (tabular - torch.tensor(tabular_scaler['mean'])) / torch.tensor(tabular_scaler['scale'])
        schnet_data_list.append(schnet_data)
        image_list.append(image)
        tabular_list.append(tabular.unsqueeze(0))
    return schnet_data_list, image_list, tabular_list

# --- 1. Attention Analysis ---
def analyze_attention_weights(model, schnet_data, image, tabular_x, edge_index, batch):
    """Analyze the attention weights in the multimodal fusion."""
    model.eval()
    with torch.no_grad():
        # Get features from each modality
        schnet_feats = model.schnet(schnet_data.z, schnet_data.pos, batch=schnet_data.batch)
        img_feats = model.resnet(image)
        gnn_feats = model.tabular_gnn(tabular_x, edge_index, batch)
        
        # Get attention weights
        attn_weights = torch.softmax(model.attn, dim=0)
        
        # Calculate weighted features
        feats = [schnet_feats, img_feats, gnn_feats]
        weighted_feats = [w * f for w, f in zip(attn_weights, feats)]
        
        # Plot attention weights
        plt.figure(figsize=VISUALIZATION_CONFIG['figure_size']['square'])
        plt.bar(['Geometric', 'Image', 'Knowledge Graph'], attn_weights.cpu().numpy(), width=VISUALIZATION_CONFIG['bar_width'], alpha=VISUALIZATION_CONFIG['alpha'])
        plt.title(f'Attention Weights for {TARGET_KEY} Prediction', fontsize=VISUALIZATION_CONFIG['font_size']['title'], fontweight=VISUALIZATION_CONFIG['font_weight']['title'])
        plt.ylabel('Attention Weight', fontsize=VISUALIZATION_CONFIG['font_size']['axis_label'])
        plt.ylim(0, 1)
        for i, v in enumerate(attn_weights.cpu().numpy()):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=VISUALIZATION_CONFIG['font_size']['annotation'])
        plt.tight_layout()
        # save_figure(os.path.join(EXPLAIN_DIR, f'attention_weights_{TARGET_KEY}.pdf'))
        plt.close()
        
        return attn_weights, weighted_feats

# --- 2. Feature Ablation Study ---
def feature_ablation_study(model, schnet_data, image, tabular_x, edge_index, batch, target_scaler):
    """Study how prediction changes when features are ablated."""
    model.eval()
    
    # Get baseline prediction
    with torch.no_grad():
        baseline_pred = model(schnet_data, image, tabular_x, edge_index, batch)
        baseline_pred_original = target_scaler['scale'][0] * baseline_pred.cpu().numpy() + target_scaler['mean'][0]
    
    # Ablate each tabular feature
    feature_names = CONFIG['tabular_keys']
    ablation_results = []
    
    for i, feature_name in enumerate(feature_names):
        # Create ablated input (zero out the feature)
        ablated_tabular = tabular_x.clone()
        ablated_tabular[0, i] = 0.0
        
        with torch.no_grad():
            ablated_pred = model(schnet_data, image, ablated_tabular, edge_index, batch)
            ablated_pred_original = target_scaler['scale'][0] * ablated_pred.cpu().numpy() + target_scaler['mean'][0]
        
        # Calculate change in prediction
        pred_change = abs(ablated_pred_original - baseline_pred_original)
        pred_change_percent = (pred_change / abs(baseline_pred_original)) * 100
        
        ablation_results.append({
            'feature': feature_name,
            'baseline_pred': baseline_pred_original[0].item(),
            'ablated_pred': ablated_pred_original[0].item(),
            'absolute_change': pred_change[0].item(),
            'percent_change': pred_change_percent[0].item()
        })
        
        # print(f"  {feature_name}: {pred_change_percent[0].item():.2f}% change")
    
    # Plot ablation results
    plt.figure(figsize=VISUALIZATION_CONFIG['figure_size']['square'])
    
    # Absolute change
    plt.subplot(2, 1, 1)
    changes = [r['absolute_change'] for r in ablation_results]
    plt.bar(feature_names, changes)
    plt.title(f'Feature Ablation: Absolute Change in {TARGET_KEY} Prediction', fontsize=VISUALIZATION_CONFIG['font_size']['title'], fontweight=VISUALIZATION_CONFIG['font_weight']['title'])
    plt.ylabel('Absolute Change', fontsize=VISUALIZATION_CONFIG['font_size']['axis_label'])
    plt.xticks(rotation=45, fontsize=VISUALIZATION_CONFIG['font_size']['tick_label'])
    
    # Percentage change
    plt.subplot(2, 1, 2)
    pct_changes = [r['percent_change'] for r in ablation_results]
    plt.bar(feature_names, pct_changes)
    plt.title(f'Feature Ablation: Percentage Change in {TARGET_KEY} Prediction', fontsize=VISUALIZATION_CONFIG['font_size']['title'], fontweight=VISUALIZATION_CONFIG['font_weight']['title'])
    plt.ylabel('Percentage Change (%)', fontsize=VISUALIZATION_CONFIG['font_size']['axis_label'])
    plt.xticks(rotation=45, fontsize=VISUALIZATION_CONFIG['font_size']['tick_label'])
    
    plt.tight_layout()
    # save_figure(os.path.join(EXPLAIN_DIR, f'feature_ablation_{TARGET_KEY}.pdf'))
    plt.close()
    
    return ablation_results

# --- 3. Gradient-based Explanations ---
def gradient_based_explanations(model, schnet_data, image, tabular_x, edge_index, batch):
    """Compute gradient-based explanations for tabular and image inputs."""
    model.eval()
    
    # Tabular gradients
    tabular_x.requires_grad_(True)
    pred = model(schnet_data, image, tabular_x, edge_index, batch)
    pred.backward()
    
    tabular_gradients = tabular_x.grad.clone()
    tabular_x.requires_grad_(False)
    
    # Image gradients
    image.requires_grad_(True)
    pred = model(schnet_data, image, tabular_x, edge_index, batch)
    pred.backward()
    
    image_gradients = image.grad.clone()
    image.requires_grad_(False)
    
    # Plot tabular gradients
    feature_names = CONFIG['tabular_keys']
    tab_grad_np = tabular_gradients.cpu().numpy().flatten()
    
    plt.figure(figsize=VISUALIZATION_CONFIG['figure_size']['square'])
    
    plt.subplot(1, 3, 1)
    plt.bar(feature_names, np.abs(tab_grad_np))
    plt.title(f'Tabular Feature Gradients for {TARGET_KEY}', fontsize=VISUALIZATION_CONFIG['font_size']['title'], fontweight=VISUALIZATION_CONFIG['font_weight']['title'])
    plt.ylabel('|Gradient|', fontsize=VISUALIZATION_CONFIG['font_size']['axis_label'])
    plt.xticks(rotation=45, fontsize=VISUALIZATION_CONFIG['font_size']['tick_label'])
    
    # Plot image gradient magnitude
    plt.subplot(1, 3, 2)
    img_grad_magnitude = torch.norm(image_gradients, dim=1).cpu().numpy()[0]
    plt.imshow(img_grad_magnitude, cmap='hot')
    plt.title('Image Gradient Magnitude', fontsize=VISUALIZATION_CONFIG['font_size']['title'], fontweight=VISUALIZATION_CONFIG['font_weight']['title'])
    plt.colorbar()
    plt.axis('off')
    
    # Plot image gradient direction
    plt.subplot(1, 3, 3)
    img_grad_mean = torch.mean(image_gradients, dim=1).cpu().numpy()[0]
    plt.imshow(img_grad_mean, cmap='RdBu_r')
    plt.title('Image Gradient Direction', fontsize=VISUALIZATION_CONFIG['font_size']['title'], fontweight=VISUALIZATION_CONFIG['font_weight']['title'])
    plt.colorbar()
    plt.axis('off')
    
    plt.tight_layout()
    # save_figure(os.path.join(EXPLAIN_DIR, f'gradient_explanations_{TARGET_KEY}.pdf'))
    plt.close()
    
    return tabular_gradients, image_gradients

# --- 4. Captum Integrated Gradients ---
def captum_integrated_gradients(model, schnet_data, image, tabular_x, edge_index, batch):
    """Compute Integrated Gradients for both tabular and image inputs."""
    model.eval()
    with torch.no_grad():
        # Tabular Integrated Gradients
        def forward_tabular(x):
            return model(schnet_data, image, x, edge_index, batch)
        ig_tabular = IntegratedGradients(forward_tabular)
        attr_tabular = ig_tabular.attribute(tabular_x, target=0)
        # Image Integrated Gradients - handle dimension issues
        def forward_image(x):
            # Ensure x has the same shape as the original image
            if x.shape != image.shape:
                x = x.view(image.shape)
            return model(schnet_data, x, tabular_x, edge_index, batch)
        try:
            ig_image = IntegratedGradients(forward_image)
            # Ensure image is 4D [1, 3, 224, 224]
            img = image
            if img.dim() == 3:
                img = img.unsqueeze(0)
            img = img.contiguous()
            if img.shape != (1, 3, 224, 224):
                img = img.view(1, 3, 224, 224)
            attr_image = ig_image.attribute(img, target=0)
        except Exception as e:
            # print(f"Warning: Could not compute image integrated gradients: {e}")
            # print("Skipping image attribution...")
            attr_image = torch.zeros_like(image)
        # Plot results
        plt.figure(figsize=VISUALIZATION_CONFIG['figure_size']['square'])
        # Tabular attributions
        plt.subplot(1, 3, 1)
        feature_names = CONFIG['tabular_keys']
        tab_attr_np = attr_tabular.detach().cpu().numpy().flatten()
        plt.bar(feature_names, tab_attr_np)
        plt.title(f'Integrated Gradients: Tabular Features', fontsize=VISUALIZATION_CONFIG['font_size']['title'], fontweight=VISUALIZATION_CONFIG['font_weight']['title'])
        plt.ylabel('Attribution', fontsize=VISUALIZATION_CONFIG['font_size']['axis_label'])
        plt.xticks(rotation=45, fontsize=VISUALIZATION_CONFIG['font_size']['tick_label'])
        # Image attributions (magnitude)
        plt.subplot(1, 3, 2)
        if torch.all(attr_image == 0):
            plt.text(0.5, 0.5, 'Image attribution\nnot available', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Integrated Gradients: Image Magnitude', fontsize=VISUALIZATION_CONFIG['font_size']['title'], fontweight=VISUALIZATION_CONFIG['font_weight']['title'])
        else:
            img_attr_magnitude = torch.norm(attr_image, dim=1).detach().cpu().numpy()[0]
            plt.imshow(img_attr_magnitude, cmap='hot')
            plt.title('Integrated Gradients: Image Magnitude', fontsize=VISUALIZATION_CONFIG['font_size']['title'], fontweight=VISUALIZATION_CONFIG['font_weight']['title'])
            plt.colorbar()
        plt.axis('off')
        # Image attributions (mean)
        plt.subplot(1, 3, 3)
        if torch.all(attr_image == 0):
            plt.text(0.5, 0.5, 'Image attribution\nnot available', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Integrated Gradients: Image Mean', fontsize=VISUALIZATION_CONFIG['font_size']['title'], fontweight=VISUALIZATION_CONFIG['font_weight']['title'])
        else:
            img_attr_mean = torch.mean(attr_image, dim=1).detach().cpu().numpy()[0]
            plt.imshow(img_attr_mean, cmap='RdBu_r')
            plt.title('Integrated Gradients: Image Mean', fontsize=VISUALIZATION_CONFIG['font_size']['title'], fontweight=VISUALIZATION_CONFIG['font_weight']['title'])
            plt.colorbar()
        plt.axis('off')
        plt.tight_layout()
        # save_figure(os.path.join(EXPLAIN_DIR, f'integrated_gradients_{TARGET_KEY}.pdf'))
        plt.close()
        return attr_tabular, attr_image

# --- 5. Model Behavior Analysis ---
def model_behavior_analysis(model, schnet_data, image, tabular_x, edge_index, batch, target_scaler):
    """Analyze how model predictions change with input perturbations."""
    model.eval()
    
    # Get baseline prediction
    with torch.no_grad():
        baseline_pred = model(schnet_data, image, tabular_x, edge_index, batch)
        baseline_pred_original = target_scaler['scale'][0] * baseline_pred.cpu().numpy() + target_scaler['mean'][0]
    
    # Perturb tabular features
    feature_names = CONFIG['tabular_keys']
    perturbation_results = []
    
    for i, feature_name in enumerate(feature_names):
        perturbations = []
        pred_changes = []
        
        # Test different perturbation levels
        for scale in [-0.5, -0.25, 0.25, 0.5]:
            perturbed_tabular = tabular_x.clone()
            perturbed_tabular[0, i] *= (1 + scale)
            
            with torch.no_grad():
                perturbed_pred = model(schnet_data, image, perturbed_tabular, edge_index, batch)
                perturbed_pred_original = target_scaler['scale'][0] * perturbed_pred.cpu().numpy() + target_scaler['mean'][0]
            
            perturbations.append(scale)
            pred_changes.append(perturbed_pred_original[0] - baseline_pred_original[0])
        
        perturbation_results.append({
            'feature': feature_name,
            'perturbations': perturbations,
            'pred_changes': pred_changes
        })
    
    # Plot perturbation analysis
    square = VISUALIZATION_CONFIG['figure_size']['square']
    fig, axes = plt.subplots(2, 3, figsize=(3 * square[0], 2 * square[1]))
    axes = axes.flatten()
    
    for i, result in enumerate(perturbation_results):
        if i < len(axes):
            ax = axes[i]
            ax.plot(result['perturbations'], result['pred_changes'], 'o-', linewidth=VISUALIZATION_CONFIG['line_width'], markersize=VISUALIZATION_CONFIG['node_size']['medium'])
            ax.axhline(y=0, color='black', linestyle='--', alpha=VISUALIZATION_CONFIG['edge_alpha'])
            ax.axvline(x=0, color='black', linestyle='--', alpha=VISUALIZATION_CONFIG['edge_alpha'])
            ax.set_title(f'{result["feature"]} Perturbation', fontsize=VISUALIZATION_CONFIG['font_size']['title'], fontweight=VISUALIZATION_CONFIG['font_weight']['title'])
            ax.set_xlabel('Perturbation Scale', fontsize=VISUALIZATION_CONFIG['font_size']['axis_label'])
            ax.set_ylabel('Prediction Change', fontsize=VISUALIZATION_CONFIG['font_size']['axis_label'])
            ax.grid(True, alpha=VISUALIZATION_CONFIG['edge_alpha'])
    
    # Hide unused subplots
    for i in range(len(perturbation_results), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    # save_figure(os.path.join(EXPLAIN_DIR, f'model_behavior_{TARGET_KEY}.pdf'))
    plt.close()
    
    return perturbation_results

# --- 6. GNNExplainer for Tabular GNN ---
def explain_gnn(model, tabular_x, batch):
    """Use gradient-based analysis to explain the tabular GNN component."""
    model.eval()
    
    # Create a simple wrapper for the GNN component
    class GNNWrapper(torch.nn.Module):
        def __init__(self, original_model):
            super().__init__()
            self.original_model = original_model
        
        def forward(self, x, edge_index, batch):
            return self.original_model.tabular_gnn(x, edge_index, batch)
    
    wrapper = GNNWrapper(model)
    
    # Use gradient-based analysis instead of GNNExplainer
    tabular_x.requires_grad_(True)
    edge_index = torch.zeros((2, 0), dtype=torch.long, device=CONFIG['device'])
    
    # Get GNN output
    gnn_output = wrapper(tabular_x, edge_index, batch)
    
    # Compute gradients - take the mean to make it scalar
    gnn_output_mean = gnn_output.mean()
    gnn_output_mean.backward()
    
    # Get feature importance from gradients
    feature_importance = torch.abs(tabular_x.grad).detach().cpu().numpy().flatten()
    tabular_x.requires_grad_(False)
    
    # print('Feature importance from GNN gradients:', feature_importance)
    
    # Plot feature importance from GNN gradients
    feature_names = CONFIG['tabular_keys']
    
    plt.figure(figsize=VISUALIZATION_CONFIG['figure_size']['square'])
    plt.bar(feature_names, feature_importance)
    plt.title(f'GNN Component Feature Importance for {TARGET_KEY}', fontsize=VISUALIZATION_CONFIG['font_size']['title'], fontweight=VISUALIZATION_CONFIG['font_weight']['title'])
    plt.xlabel('Features', fontsize=VISUALIZATION_CONFIG['font_size']['axis_label'])
    plt.ylabel('Gradient Magnitude', fontsize=VISUALIZATION_CONFIG['font_size']['axis_label'])
    plt.xticks(rotation=45, fontsize=VISUALIZATION_CONFIG['font_size']['tick_label'])
    plt.tight_layout()
    # save_figure(os.path.join(EXPLAIN_DIR, f'gnn_explainer_{TARGET_KEY}.pdf'))
    plt.close()
    
    return feature_importance

# --- 7. Comprehensive Summary Report ---
def generate_summary_report(attention_weights, ablation_results, tabular_gradients, 
                          perturbation_results, gnn_importance):
    """Generate a comprehensive summary report of all explainability methods."""
    fig, axes = plt.subplots(2, 3, figsize=(3 * VISUALIZATION_CONFIG['figure_size']['square'][0], 2 * VISUALIZATION_CONFIG['figure_size']['square'][1]))
    # 1. Attention weights
    attn_vals = attention_weights.cpu().numpy()
    attn_labels = ['Geometric', 'Image', 'Knowledge Graph']
    axes[0, 0].bar(attn_labels, attn_vals)
    axes[0, 0].set_title('(A)', fontsize=VISUALIZATION_CONFIG['font_size']['title'], fontweight=VISUALIZATION_CONFIG['font_weight']['title'])
    axes[0, 0].set_ylabel('Weight', fontsize=VISUALIZATION_CONFIG['font_size']['axis_label'])
    attn_offset = 0.02 * max(attn_vals) if len(attn_vals) > 0 else 0.01
    attn_sum = np.sum(attn_vals)
    for i, v in enumerate(attn_vals):
        percent = 100 * v / attn_sum if attn_sum > 0 else 0
        axes[0, 0].text(i, v + attn_offset, f'{v:.2f}\n{percent:.0f}%', ha='center', va='bottom', fontsize=VISUALIZATION_CONFIG['font_size']['annotation'])
    # 2. Feature ablation (all features, %)
    ablation_importance = [(r['feature'], r['percent_change']) for r in ablation_results]
    ablation_importance.sort(key=lambda x: x[1], reverse=True)
    ablation_vals = [f[1] for f in ablation_importance]
    ablation_labels = [f[0] for f in ablation_importance]
    axes[0, 1].bar(ablation_labels, ablation_vals)
    axes[0, 1].set_title('(B)', fontsize=VISUALIZATION_CONFIG['font_size']['title'], fontweight=VISUALIZATION_CONFIG['font_weight']['title'])
    axes[0, 1].set_ylabel('Prediction Change (%)', fontsize=VISUALIZATION_CONFIG['font_size']['axis_label'])
    axes[0, 1].tick_params(axis='x', rotation=45, labelsize=VISUALIZATION_CONFIG['font_size']['tick_label'])
    ablation_offset = 0.02 * max(ablation_vals) if len(ablation_vals) > 0 else 0.5
    for i, v in enumerate(ablation_vals):
        axes[0, 1].text(i, v + ablation_offset, f'{v:.1f}%', ha='center', va='bottom', fontsize=VISUALIZATION_CONFIG['font_size']['annotation'])
    # 3. Gradient magnitudes (% of max)
    grad_magnitudes = np.abs(tabular_gradients.cpu().numpy().flatten())
    grad_percent = 100 * grad_magnitudes / (np.max(grad_magnitudes) if np.max(grad_magnitudes) > 0 else 1)
    feature_names = CONFIG['tabular_keys']
    axes[0, 2].bar(feature_names, grad_percent)
    axes[0, 2].set_title('(C)', fontsize=VISUALIZATION_CONFIG['font_size']['title'], fontweight=VISUALIZATION_CONFIG['font_weight']['title'])
    axes[0, 2].set_ylabel('|Gradient| (% of max)', fontsize=VISUALIZATION_CONFIG['font_size']['axis_label'])
    axes[0, 2].tick_params(axis='x', rotation=45, labelsize=VISUALIZATION_CONFIG['font_size']['tick_label'])
    grad_offset = 0.02 * max(grad_percent) if len(grad_percent) > 0 else 0.5
    for i, v in enumerate(grad_percent):
        axes[0, 2].text(i, v + grad_offset, f'{v:.1f}%', ha='center', va='bottom', fontsize=VISUALIZATION_CONFIG['font_size']['annotation'])
    # 4. GNNExplainer importance (% of max)
    gnn_percent = 100 * gnn_importance / (np.max(gnn_importance) if np.max(gnn_importance) > 0 else 1)
    axes[1, 0].bar(feature_names, gnn_percent)
    axes[1, 0].set_title('(D)', fontsize=VISUALIZATION_CONFIG['font_size']['title'], fontweight=VISUALIZATION_CONFIG['font_weight']['title'])
    axes[1, 0].set_ylabel('Importance (% of max)', fontsize=VISUALIZATION_CONFIG['font_size']['axis_label'])
    axes[1, 0].tick_params(axis='x', rotation=45, labelsize=VISUALIZATION_CONFIG['font_size']['tick_label'])
    gnn_offset = 0.02 * max(gnn_percent) if len(gnn_percent) > 0 else 0.5
    for i, v in enumerate(gnn_percent):
        axes[1, 0].text(i, v + gnn_offset, f'{v:.1f}%', ha='center', va='bottom', fontsize=VISUALIZATION_CONFIG['font_size']['annotation'])
    # 5. Perturbation sensitivity (% of max)
    sensitivity_scores = []
    for result in perturbation_results:
        sensitivity = np.mean(np.abs(result['pred_changes']))
        sensitivity_scores.append(sensitivity)
    sensitivity_percent = 100 * np.array(sensitivity_scores) / (np.max(sensitivity_scores) if np.max(sensitivity_scores) > 0 else 1)
    axes[1, 1].bar(feature_names, sensitivity_percent)
    axes[1, 1].set_title('(E)', fontsize=VISUALIZATION_CONFIG['font_size']['title'], fontweight=VISUALIZATION_CONFIG['font_weight']['title'])
    axes[1, 1].set_ylabel('Sensitivity (% of max)', fontsize=VISUALIZATION_CONFIG['font_size']['axis_label'])
    axes[1, 1].tick_params(axis='x', rotation=45, labelsize=VISUALIZATION_CONFIG['font_size']['tick_label'])
    sens_offset = 0.02 * max(sensitivity_percent) if len(sensitivity_percent) > 0 else 0.5
    for i, v in enumerate(sensitivity_percent):
        axes[1, 1].text(i, v + sens_offset, f'{v:.1f}%', ha='center', va='bottom', fontsize=VISUALIZATION_CONFIG['font_size']['annotation'])
    # 6. Consensus ranking (% of max)
    consensus_scores = np.zeros(len(feature_names))
    if ablation_results:
        ablation_scores = np.array([r['percent_change'] for r in ablation_results])
        consensus_scores += ablation_scores / np.max(ablation_scores)
    if len(grad_percent) > 0:
        consensus_scores += grad_percent / 100
    if len(gnn_percent) > 0:
        consensus_scores += gnn_percent / 100
    if len(sensitivity_percent) > 0:
        consensus_scores += sensitivity_percent / 100
    consensus_percent = 100 * consensus_scores / np.max(consensus_scores) if np.max(consensus_scores) > 0 else consensus_scores
    sorted_indices = np.argsort(consensus_percent)[::-1]
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_scores = [consensus_percent[i] for i in sorted_indices]
    axes[1, 2].bar(sorted_features, sorted_scores)
    axes[1, 2].set_title('(F)', fontsize=VISUALIZATION_CONFIG['font_size']['title'], fontweight=VISUALIZATION_CONFIG['font_weight']['title'])
    axes[1, 2].set_ylabel('Consensus Score (%)', fontsize=VISUALIZATION_CONFIG['font_size']['axis_label'])
    axes[1, 2].tick_params(axis='x', rotation=45, labelsize=VISUALIZATION_CONFIG['font_size']['tick_label'])
    cons_offset = 0.02 * max(sorted_scores) if len(sorted_scores) > 0 else 0.5
    for i, v in enumerate(sorted_scores):
        axes[1, 2].text(i, v + cons_offset, f'{v:.1f}%', ha='center', va='bottom', fontsize=VISUALIZATION_CONFIG['font_size']['annotation'])
    plt.tight_layout()
    save_figure(os.path.join(EXPLAIN_DIR, f'comprehensive_summary_{TARGET_KEY}.pdf'))
    plt.close()
    # print("\nConsensus Feature Importance Ranking:")
    # for i, (feature, score) in enumerate(zip(sorted_features, sorted_scores)):
    #     print(f"  {i+1}. {feature}: {score:.1f}%")

if __name__ == '__main__':
    # print(f"Running comprehensive explainability analysis for all R-H2 clusters - {TARGET_KEY}")
    # Create explainability folder if it doesn't exist
    EXPLAIN_DIR = 'explainability'
    os.makedirs(EXPLAIN_DIR, exist_ok=True)
    # Load data and model
    kg = load_knowledge_graph('new_data/knowledge_graph.json')
    # Find all R ids with -H2
    r_ids = [node['id'] for node in kg if node['id'].endswith('-H2')]
    # print(f"Averaging over R values: {r_ids}")
    # Prepare containers for averaging across all R
    attn_weights_allR = []
    ablation_results_allR = []
    tabular_gradients_allR = []
    perturbation_results_allR = []
    gnn_importance_allR = []
    for LEAVE_OUT_ID in r_ids:
        model, tabular_scaler, target_scaler = load_model_and_scalers(LEAVE_OUT_ID, TARGET_KEY)
        schnet_data_list, image_list, tabular_list = prepare_all_test_samples(kg, LEAVE_OUT_ID, TARGET_KEY, tabular_scaler)
        # Per-R containers
        attn_weights_all = []
        ablation_results_all = []
        tabular_gradients_all = []
        perturbation_results_all = []
        gnn_importance_all = []
        for schnet_data, image, tabular_x in zip(schnet_data_list, image_list, tabular_list):
            # Move to device
            schnet_data = schnet_data.to(CONFIG['device'])
            image = image.unsqueeze(0).to(CONFIG['device'])
            tabular_x = tabular_x.to(CONFIG['device'])
            batch = torch.zeros(tabular_x.shape[0], dtype=torch.long, device=CONFIG['device'])
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=CONFIG['device'])
            # 1. Attention analysis
            attn_weights, weighted_feats = analyze_attention_weights(model, schnet_data, image, tabular_x, edge_index, batch)
            attn_weights_all.append(attn_weights.cpu().numpy())
            # 2. Feature ablation study
            ablation_results = feature_ablation_study(model, schnet_data, image, tabular_x, edge_index, batch, target_scaler)
            ablation_results_all.append(ablation_results)
            # 3. Gradient-based explanations
            tabular_gradients, image_gradients = gradient_based_explanations(model, schnet_data, image, tabular_x, edge_index, batch)
            tabular_gradients_all.append(tabular_gradients.cpu().numpy())
            # 4. Captum Integrated Gradients
            attr_tabular, attr_image = captum_integrated_gradients(model, schnet_data, image, tabular_x, edge_index, batch)
            # 5. Model behavior analysis
            perturbation_results = model_behavior_analysis(model, schnet_data, image, tabular_x, edge_index, batch, target_scaler)
            perturbation_results_all.append(perturbation_results)
            # 6. GNNExplainer
            gnn_importance = explain_gnn(model, tabular_x, batch)
            gnn_importance_all.append(gnn_importance)
        # Average per-R
        attn_weights_avg = np.mean(attn_weights_all, axis=0)
        ablation_avg = []
        for i in range(len(CONFIG['tabular_keys'])):
            percent_changes = [abl[i]['percent_change'] for abl in ablation_results_all]
            absolute_changes = [abl[i]['absolute_change'] for abl in ablation_results_all]
            ablation_avg.append({
                'feature': CONFIG['tabular_keys'][i],
                'percent_change': np.mean(percent_changes),
                'absolute_change': np.mean(absolute_changes)
            })
        tabular_gradients_avg = np.mean(tabular_gradients_all, axis=0)
        gnn_importance_avg = np.mean(gnn_importance_all, axis=0)
        perturbation_avg = []
        for i in range(len(CONFIG['tabular_keys'])):
            all_pred_changes = [np.array([p[i]['pred_changes'] for p in perturbation_results_all]).mean(axis=0)]
            perturbation_avg.append({
                'feature': CONFIG['tabular_keys'][i],
                'perturbations': perturbation_results_all[0][i]['perturbations'],
                'pred_changes': all_pred_changes[0]
            })
        # Collect for all R
        attn_weights_allR.append(attn_weights_avg)
        ablation_results_allR.append(ablation_avg)
        tabular_gradients_allR.append(tabular_gradients_avg)
        perturbation_results_allR.append(perturbation_avg)
        gnn_importance_allR.append(gnn_importance_avg)
    # Final average across all R
    attn_weights_final = np.mean(attn_weights_allR, axis=0)
    ablation_final = []
    for i in range(len(CONFIG['tabular_keys'])):
        percent_changes = [abl[i]['percent_change'] for abl in ablation_results_allR]
        absolute_changes = [abl[i]['absolute_change'] for abl in ablation_results_allR]
        ablation_final.append({
            'feature': CONFIG['tabular_keys'][i],
            'percent_change': np.mean(percent_changes),
            'absolute_change': np.mean(absolute_changes)
        })
    tabular_gradients_final = np.mean(tabular_gradients_allR, axis=0)
    gnn_importance_final = np.mean(gnn_importance_allR, axis=0)
    perturbation_final = []
    for i in range(len(CONFIG['tabular_keys'])):
        all_pred_changes = [np.array([p[i]['pred_changes'] for p in perturbation_results_allR]).mean(axis=0)]
        perturbation_final.append({
            'feature': CONFIG['tabular_keys'][i],
            'perturbations': perturbation_results_allR[0][i]['perturbations'],
            'pred_changes': all_pred_changes[0]
        })
    # 7. Generate comprehensive summary
    generate_summary_report(
        torch.tensor(attn_weights_final),
        ablation_final,
        torch.tensor(tabular_gradients_final),
        perturbation_final,
        gnn_importance_final
    )
    print("\n" + "="*60)
    print("Analysis complete! Check the generated PDF files for visualizations.")
    print("="*60) 