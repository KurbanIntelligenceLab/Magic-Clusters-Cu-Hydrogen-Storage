# ------------------- CONFIG -------------------
CONFIG = {
    "seeds": [42, 123],
    "batch_size": 2,
    "epochs": 500,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "scheduler": {"T_0": 100, "T_mult": 2, "eta_min": 1e-6},
    "leave_out_ids": ["R7", "R8", "R9", "R10", "R7-H2", "R8-H2", "R9-H2", "R10-H2"],
    "tabular_keys": ["Cu", "Ef_f", "Ef_t", "HOMO", "LUMO", "Eg", "H2", "Cu-H2"],
    "target_keys": ["HOMO", "LUMO", "Eg", "Ef_t", "Ef_f"],
    "model_params": {
        "tabular_dim": 8,
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
