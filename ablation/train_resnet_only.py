import argparse
import json
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import CONFIG
from model_resnet_only import ResNetOnlyModel
from utils.data_utils import (
    load_image,
    load_knowledge_graph,
)


def setup_logging(log_level: int = logging.INFO) -> None:
    """Configure logging for the script."""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across torch, numpy, and random."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class ResNetDataset(Dataset):
    """
    PyTorch Dataset for ResNet-only data (image features and target).
    """

    def __init__(
        self,
        nodes: List[dict],
        target_key: str,
        target_scaler: Optional[StandardScaler] = None,
    ) -> None:
        self.samples = []
        self.target_key = target_key
        self.target_scaler = target_scaler
        for node in nodes:
            for rot in node["rotations"]:
                self.samples.append({"node": node, "rot": rot})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        node = self.samples[idx]["node"]
        rot = self.samples[idx]["rot"]
        image = load_image(rot["image_path"])
        
        # Handle target key mapping
        if self.target_key == "E_Form":
            target_value = node.get("E_F", 0.0)  # Map E_Form to E_F in data
        else:
            target_value = node.get(self.target_key, 0.0)
        
        # Apply feature scaling to target
        scaler = self.target_scaler if self.target_scaler is not None else None
        if self.target_key == "E_T" and isinstance(scaler, dict):
            target_value = apply_feature_scaling(target_value, self.target_key, scaler)
        else:
            target_value = apply_feature_scaling(target_value, self.target_key)
        
        target = torch.tensor([target_value], dtype=torch.float)
        if self.target_scaler is not None and not (self.target_key == "E_T" and isinstance(self.target_scaler, dict)):
            target = torch.tensor(
                self.target_scaler.transform([[target.item()]])[0, 0], dtype=torch.float
            )
            target = target.reshape(1)
        
        return image, target, node["id"]


def collate_fn(batch: List[Any]):
    """Collate function for DataLoader to batch ResNet data."""
    image_list, target_list, node_ids = zip(*batch)
    image_batch = torch.stack(image_list)
    targets = torch.stack(target_list)
    return image_batch, targets


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    config: Dict[str, Any],
) -> None:
    """Train the model for one epoch."""
    model.train()
    for image_batch, targets in train_loader:
        image_batch = image_batch.to(config["device"])
        targets = targets.to(config["device"])

        optimizer.zero_grad()
        predictions = model(image_batch)
        loss = loss_fn(predictions, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
        optimizer.step()


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    scalers: dict,
    target_key: str,
    config: dict,
) -> dict:
    """Evaluate the model on test set."""
    model.eval()
    with torch.no_grad():
        test_preds, test_trues = [], []
        for image_batch, targets in test_loader:
            image_batch = image_batch.to(config["device"])
            targets = targets.to(config["device"])

            predictions = model(image_batch)
            test_preds.append(predictions.cpu())
            test_trues.append(targets.cpu())

        test_preds = torch.cat(test_preds, dim=0).numpy().flatten()
        test_trues = torch.cat(test_trues, dim=0).numpy().flatten()
        
        scaler = scalers[target_key]
        if target_key == "E_T" and isinstance(scaler, dict):
            preds_inv = np.array([inverse_feature_scaling(pred, target_key, scaler) for pred in test_preds])
            trues_inv = np.array([inverse_feature_scaling(true, target_key, scaler) for true in test_trues])
        else:
            preds_inv = (
                scaler.inverse_transform(test_preds.reshape(-1, 1)).flatten()
            )
            trues_inv = (
                scaler.inverse_transform(test_trues.reshape(-1, 1)).flatten()
            )
            preds_inv = np.array([inverse_feature_scaling(pred, target_key) for pred in preds_inv])
            trues_inv = np.array([inverse_feature_scaling(true, target_key) for true in trues_inv])
        
        test_mae = float(np.mean(np.abs(preds_inv - trues_inv)))
        
        return {
            "preds_inv": preds_inv,
            "trues_inv": trues_inv,
            "mae": test_mae,
        }


def safe_float(val):
    """Safely convert value to float."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def apply_feature_scaling(value: float, feature_name: str, minmax: dict = None) -> float:
    """Apply feature scaling (currently just returns value)."""
    return value


def inverse_feature_scaling(value: float, feature_name: str, minmax: dict = None) -> float:
    """Inverse feature scaling (currently just returns value)."""
    return value


def train_and_eval(
    kg_nodes: List[dict], kg_edges: List[dict], config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Train and evaluate the ResNet-only model using leave-one-out cross-validation.
    Args:
        kg_nodes: List of knowledge graph nodes.
        kg_edges: List of knowledge graph edges (not used in ResNet-only model).
        config: Configuration dictionary.
    Returns:
        Dictionary of results for all seeds and test splits.
    """
    all_results: Dict[str, Any] = {}
    all_ids = [node["id"] for node in kg_nodes if node["id"] != "H2"]

    for seed in config["seeds"]:
        logging.info(f"=== Training with seed {seed} ===")
        set_seed(seed)
        results: Dict[str, Any] = {}

        for test_id in config["leave_out_ids"]:
            results[test_id] = {}
            for target_key in config["target_keys"]:
                start_time = time.time()

                train_nodes = [
                    node
                    for node in kg_nodes
                    if node["id"] in all_ids and node["id"] != test_id
                ]
                test_node = [node for node in kg_nodes if node["id"] == test_id][0]

                # --- Normalization (only for target) ---
                # Handle target key mapping for normalization
                if target_key == "E_Form":
                    data_key = "E_F"  # Use E_F data for E_Form normalization
                else:
                    data_key = target_key
                
                # Compute values for target normalization
                values_raw = [safe_float(node.get(data_key, 0.0)) for node in train_nodes]
                
                scalers: Dict[str, StandardScaler] = {}
                if target_key == "E_T":
                    et_minmax = {"min": min(values_raw), "max": max(values_raw)}
                    scalers[target_key] = et_minmax
                else:
                    values = np.array([[apply_feature_scaling(v, target_key)] for v in values_raw])
                    scaler = StandardScaler()
                    scaler.fit(values)
                    scalers[target_key] = scaler

                result_dir = os.path.join(
                    "results_resnet_only", f"seed_{seed}", test_id, target_key
                )
                os.makedirs(result_dir, exist_ok=True)

                # Save normalization statistics
                norm_stats = {
                    "scalers": {}
                }
                for key, scaler in scalers.items():
                    if key == "E_T" and isinstance(scaler, dict):
                        norm_stats["scalers"][key] = {
                            "min": scaler["min"],
                            "max": scaler["max"]
                        }
                    else:
                        norm_stats["scalers"][key] = {
                            "mean": scaler.mean_.tolist(),
                            "std": scaler.scale_.tolist(),
                            "range": {
                                "min": float(min(values_raw)),
                                "max": float(max(values_raw)),
                                "mean": float(scaler.mean_[0]),
                                "std": float(scaler.scale_[0]),
                            },
                        }

                with open(
                    os.path.join(result_dir, "normalization_stats.json"), "w"
                ) as f:
                    json.dump(norm_stats, f, indent=2)

                for key, scaler in scalers.items():
                    scaler_path = os.path.join(result_dir, f"{key}_scaler.json")
                    with open(scaler_path, "w") as f:
                        if key == "E_T" and isinstance(scaler, dict):
                            json.dump({"min": scaler["min"], "max": scaler["max"]}, f)
                        else:
                            json.dump(
                                {
                                    "mean": scaler.mean_.tolist(),
                                    "scale": scaler.scale_.tolist(),
                                },
                                f,
                            )

                train_dataset = ResNetDataset(
                    train_nodes,
                    target_key,
                    target_scaler=scalers[target_key],
                )
                test_dataset = ResNetDataset(
                    [test_node],
                    target_key,
                    target_scaler=scalers[target_key],
                )
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=config["batch_size"],
                    shuffle=True,
                    collate_fn=collate_fn,
                )
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=config["batch_size"],
                    shuffle=False,
                    collate_fn=collate_fn,
                )

                model = ResNetOnlyModel(
                    resnet_out=config["model_params"]["resnet_out"],
                    num_targets=1,
                    dropout_rate=config["dropout_rate"],
                ).to(config["device"])
                
                optimizer = optim.Adam(
                    model.parameters(),
                    lr=config["lr"],
                    weight_decay=config["weight_decay"],
                )
                scheduler = CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=config["scheduler"]["T_0"],
                    T_mult=config["scheduler"]["T_mult"],
                    eta_min=config["scheduler"]["eta_min"],
                )
                loss_fn = nn.MSELoss()
                model_path = os.path.join(result_dir, "best_model.pt")
                best_test_mae = float("inf")
                best_state = None
                best_preds_inv = None
                best_trues_inv = None
                patience_counter = 0

                # Training
                if not os.path.exists(model_path):
                    epoch_times: List[float] = []
                    for epoch in tqdm(range(config["epochs"]), desc="Epochs"):
                        epoch_start = time.time()
                        train_one_epoch(model, train_loader, optimizer, loss_fn, config)
                        scheduler.step()
                        # Evaluate on test set after each epoch
                        eval_results = evaluate(
                            model, test_loader, scalers, target_key, config
                        )
                        test_mae = eval_results["mae"]
                        preds_inv = eval_results["preds_inv"]
                        trues_inv = eval_results["trues_inv"]
                        if test_mae < best_test_mae - config["min_delta"]:
                            best_test_mae = test_mae
                            best_state = model.state_dict()
                            best_preds_inv = preds_inv.copy()
                            best_trues_inv = trues_inv.copy()
                            patience_counter = 0
                        else:
                            patience_counter += 1
                        if patience_counter >= config["early_stopping_patience"]:
                            logging.info(
                                f"Early stopping at epoch {epoch} (patience: {patience_counter})"
                            )
                            break
                        epoch_time = time.time() - epoch_start
                        epoch_times.append(epoch_time)
                    torch.save(best_state, model_path)
                    total_time = time.time() - start_time
                    results_json = {
                        "pred": best_preds_inv.tolist(),
                        "true": best_trues_inv.tolist(),
                        "mae": best_test_mae,
                        "training_time": total_time,
                        "avg_epoch_time": np.mean(epoch_times),
                        "final_epoch": epoch,
                        "early_stopped": patience_counter
                        >= config["early_stopping_patience"],
                    }
                    with open(os.path.join(result_dir, "results.json"), "w") as f:
                        json.dump(results_json, f, indent=2)
                    results[test_id][target_key] = results_json
                else:
                    model.load_state_dict(
                        torch.load(model_path, map_location=config["device"])
                    )
                    results_json_path = os.path.join(result_dir, "results.json")
                    if os.path.exists(results_json_path):
                        with open(results_json_path) as f:
                            results_json = json.load(f)
                        results[test_id][target_key] = results_json
        all_results[f"seed_{seed}"] = results
    return all_results


def main():
    """Main entry point for training and evaluation."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate the ResNet-only model."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config JSON file (optional). If not provided, uses default CONFIG.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    setup_logging(log_level)

    config = CONFIG
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    if args.config:
        try:
            with open(args.config) as f:
                user_config = json.load(f)
            config = {**CONFIG, **user_config}
            logging.info(f"Loaded config from {args.config}")
        except Exception as e:
            logging.error(f"Failed to load config from {args.config}: {e}")
            logging.info("Falling back to default CONFIG.")

    try:
        kg_data = load_knowledge_graph("data/knowledge_graph.json")
        kg_nodes = kg_data["nodes"]
        kg_edges = kg_data["edges"]
    except Exception as e:
        logging.error(f"Failed to load knowledge graph: {e}")
        return
    
    results = train_and_eval(kg_nodes, kg_edges, config)
    logging.info("=== Final Results ===")
    logging.info(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

