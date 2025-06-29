# ------------------- CONFIG -------------------
CONFIG = {
    "seeds": [30, 40],
    "batch_size": 2,
    "epochs": 500,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "scheduler": {"T_0": 100, "T_mult": 2, "eta_min": 1e-6},
    "leave_out_ids": ["R7-H2", "R8-H2", "R9-H2", "R10-H2"],
    "tabular_keys": ["E_H", "E_L", "E_g", "E_f", "E_T", "E_F", "d_Cu-H", "N_cu", "N_h"],
    "target_keys": ["E_H", "E_L", "E_g", "E_f", "E_T", "E_F", "d_Cu-H"],
    "model_params": {
        "tabular_dim": 9,
        "gnn_hidden": 64,
        "gnn_out": 32,
        "schnet_out": 32,
        "resnet_out": 512,
        "fusion_dim": 32,
        "num_targets": 1,
    },
    # Regularization parameters
    "dropout_rate": 0.5,
    "early_stopping_patience": 300,
    "min_delta": 5e-4,
    "gradient_clip": 1.0,
    "track_modalities": True,
    "modality_names": ["Geometric", "Image", "Tabular"],
}
