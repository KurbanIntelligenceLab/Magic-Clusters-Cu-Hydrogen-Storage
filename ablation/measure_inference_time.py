import argparse
import json
import logging
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import CONFIG
from model_tabular_only import TabularOnlyModel
from train_tabular_only import TabularDataset, collate_fn
from utils.data_utils import load_knowledge_graph


def setup_logging(log_level: int = logging.INFO) -> None:
    """Configure logging for the script."""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_model_and_scalers(
    model_path: str,
    scaler_dir: str,
    config: Dict,
    target_key: str,
) -> tuple:
    """Load trained model and scalers."""
    # Load model
    model = TabularOnlyModel(
        tabular_dim=len(config["tabular_keys"]),
        num_targets=1,
        dropout_rate=config["dropout_rate"],
    ).to(config["device"])
    
    model.load_state_dict(torch.load(model_path, map_location=config["device"]))
    model.eval()
    
    # Load scalers
    scalers = {}
    for key in config["tabular_keys"] + [target_key]:
        scaler_path = os.path.join(scaler_dir, f"{key}_scaler.json")
        if os.path.exists(scaler_path):
            with open(scaler_path) as f:
                scaler_data = json.load(f)
                if key == "E_T" and "min" in scaler_data:
                    scalers[key] = scaler_data
                else:
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    scaler.mean_ = np.array(scaler_data["mean"])
                    scaler.scale_ = np.array(scaler_data["scale"])
                    scalers[key] = scaler
    
    return model, scalers


def measure_inference_time(
    model: nn.Module,
    data_loader: DataLoader,
    config: Dict,
    num_warmup: int = 10,
    num_runs: int = 100,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Measure inference time for the model.
    
    Args:
        model: The model to measure
        data_loader: DataLoader with test data
        config: Configuration dictionary
        num_warmup: Number of warmup runs
        num_runs: Number of inference runs for statistics
        device: Device to run on
    
    Returns:
        Dictionary with timing statistics
    """
    model.eval()
    
    # Get a batch of data
    tabular_x, _ = next(iter(data_loader))
    tabular_x = tabular_x.to(device)
    
    # Warmup runs
    logging.info(f"Running {num_warmup} warmup iterations...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(tabular_x)
    
    # Synchronize if using GPU
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    
    # Measure inference time
    logging.info(f"Measuring inference time over {num_runs} runs...")
    times = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            _ = model(tabular_x)
            
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to milliseconds
    
    times = np.array(times)
    
    stats = {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "median_ms": float(np.median(times)),
        "p25_ms": float(np.percentile(times, 25)),
        "p75_ms": float(np.percentile(times, 75)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)),
        "throughput_samples_per_sec": float(1.0 / (np.mean(times) / 1000.0) * tabular_x.shape[0]),
    }
    
    return stats


def measure_batch_inference_time(
    model: nn.Module,
    data_loader: DataLoader,
    config: Dict,
    num_warmup: int = 5,
    num_runs: int = 50,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Measure inference time over entire dataset (batch processing).
    
    Args:
        model: The model to measure
        data_loader: DataLoader with test data
        config: Configuration dictionary
        num_warmup: Number of warmup runs
        num_runs: Number of full dataset passes
        device: Device to run on
    
    Returns:
        Dictionary with timing statistics
    """
    model.eval()
    
    # Warmup
    logging.info(f"Running {num_warmup} warmup passes...")
    with torch.no_grad():
        for _ in range(num_warmup):
            for tabular_x, _ in data_loader:
                tabular_x = tabular_x.to(device)
                _ = model(tabular_x)
    
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    
    # Measure full dataset inference time
    logging.info(f"Measuring full dataset inference time over {num_runs} passes...")
    times = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            for tabular_x, _ in data_loader:
                tabular_x = tabular_x.to(device)
                _ = model(tabular_x)
            
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to milliseconds
    
    times = np.array(times)
    num_samples = len(data_loader.dataset)
    
    stats = {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "median_ms": float(np.median(times)),
        "mean_per_sample_ms": float(np.mean(times) / num_samples),
        "throughput_samples_per_sec": float(num_samples / (np.mean(times) / 1000.0)),
    }
    
    return stats


def main():
    """Main entry point for inference time measurement."""
    parser = argparse.ArgumentParser(
        description="Measure inference time for tabular-only model."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--scaler-dir",
        type=str,
        required=True,
        help="Directory containing scaler JSON files",
    )
    parser.add_argument(
        "--target-key",
        type=str,
        default="E_H",
        help="Target key to use (default: E_H)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for inference (default: use config batch_size)",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=10,
        help="Number of warmup runs (default: 10)",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=100,
        help="Number of inference runs for statistics (default: 100)",
    )
    parser.add_argument(
        "--use-dummy-data",
        action="store_true",
        help="Use dummy data instead of loading from knowledge graph",
    )
    parser.add_argument(
        "--dummy-batch-size",
        type=int,
        default=1,
        help="Batch size for dummy data (default: 1)",
    )
    parser.add_argument(
        "--measure-batch",
        action="store_true",
        help="Also measure full dataset inference time",
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

    config = CONFIG.copy()
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    if args.batch_size:
        config["batch_size"] = args.batch_size

    logging.info(f"Using device: {config['device']}")
    logging.info(f"Model path: {args.model_path}")
    logging.info(f"Scaler directory: {args.scaler_dir}")
    logging.info(f"Target key: {args.target_key}")

    # Load model and scalers
    try:
        model, scalers = load_model_and_scalers(
            args.model_path, args.scaler_dir, config, args.target_key
        )
        logging.info("Model and scalers loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load model/scalers: {e}")
        return

    # Prepare data
    if args.use_dummy_data:
        logging.info("Using dummy data for inference timing")
        tabular_dim = len(config["tabular_keys"])
        dummy_data = torch.randn(args.dummy_batch_size, tabular_dim)
        data_loader = DataLoader(
            [(dummy_data[i], torch.tensor([0.0])) for i in range(args.dummy_batch_size)],
            batch_size=args.dummy_batch_size,
            shuffle=False,
        )
    else:
        try:
            kg_data = load_knowledge_graph("data/knowledge_graph.json")
            kg_nodes = kg_data["nodes"]
            
            # Use first available node for testing
            test_node = kg_nodes[0] if kg_nodes else None
            if not test_node:
                logging.error("No nodes found in knowledge graph")
                return
            
            # Create filtered scalers dict
            filtered_scalers = {
                k: scalers[k] for k in config["tabular_keys"] if k in scalers
            }
            
            test_dataset = TabularDataset(
                [test_node],
                config["tabular_keys"],
                args.target_key,
                tabular_scaler=filtered_scalers,
                target_scaler=scalers.get(args.target_key),
                exclude_target_for_ids=[],
                target_keys=config["target_keys"],
            )
            data_loader = DataLoader(
                test_dataset,
                batch_size=config["batch_size"],
                shuffle=False,
                collate_fn=collate_fn,
            )
            logging.info(f"Loaded test data: {len(test_dataset)} samples")
        except Exception as e:
            logging.error(f"Failed to load data: {e}")
            return

    # Measure single batch inference time
    logging.info("\n" + "=" * 60)
    logging.info("Measuring single batch inference time...")
    logging.info("=" * 60)
    
    single_batch_stats = measure_inference_time(
        model,
        data_loader,
        config,
        num_warmup=args.num_warmup,
        num_runs=args.num_runs,
        device=config["device"],
    )
    
    logging.info("\nSingle Batch Inference Statistics:")
    logging.info(f"  Mean:     {single_batch_stats['mean_ms']:.4f} ms")
    logging.info(f"  Std:      {single_batch_stats['std_ms']:.4f} ms")
    logging.info(f"  Min:      {single_batch_stats['min_ms']:.4f} ms")
    logging.info(f"  Max:      {single_batch_stats['max_ms']:.4f} ms")
    logging.info(f"  Median:   {single_batch_stats['median_ms']:.4f} ms")
    logging.info(f"  P25:      {single_batch_stats['p25_ms']:.4f} ms")
    logging.info(f"  P75:      {single_batch_stats['p75_ms']:.4f} ms")
    logging.info(f"  P95:      {single_batch_stats['p95_ms']:.4f} ms")
    logging.info(f"  P99:      {single_batch_stats['p99_ms']:.4f} ms")
    logging.info(
        f"  Throughput: {single_batch_stats['throughput_samples_per_sec']:.2f} samples/sec"
    )

    # Measure full dataset inference time if requested
    if args.measure_batch:
        logging.info("\n" + "=" * 60)
        logging.info("Measuring full dataset inference time...")
        logging.info("=" * 60)
        
        batch_stats = measure_batch_inference_time(
            model,
            data_loader,
            config,
            num_warmup=5,
            num_runs=min(50, args.num_runs // 2),
            device=config["device"],
        )
        
        logging.info("\nFull Dataset Inference Statistics:")
        logging.info(f"  Mean (total):     {batch_stats['mean_ms']:.4f} ms")
        logging.info(f"  Mean (per sample): {batch_stats['mean_per_sample_ms']:.4f} ms")
        logging.info(f"  Std:              {batch_stats['std_ms']:.4f} ms")
        logging.info(f"  Min:               {batch_stats['min_ms']:.4f} ms")
        logging.info(f"  Max:               {batch_stats['max_ms']:.4f} ms")
        logging.info(f"  Median:            {batch_stats['median_ms']:.4f} ms")
        logging.info(
            f"  Throughput:        {batch_stats['throughput_samples_per_sec']:.2f} samples/sec"
        )
        
        # Save results
        results = {
            "single_batch": single_batch_stats,
            "full_dataset": batch_stats,
            "config": {
                "device": config["device"],
                "batch_size": config["batch_size"],
                "target_key": args.target_key,
                "num_warmup": args.num_warmup,
                "num_runs": args.num_runs,
            },
        }
    else:
        results = {
            "single_batch": single_batch_stats,
            "config": {
                "device": config["device"],
                "batch_size": config["batch_size"],
                "target_key": args.target_key,
                "num_warmup": args.num_warmup,
                "num_runs": args.num_runs,
            },
        }

    # Save results to JSON
    output_path = os.path.join(
        os.path.dirname(args.model_path), "inference_time_results.json"
    )
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

