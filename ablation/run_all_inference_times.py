#!/usr/bin/env python3
"""
Batch script to run inference time measurements for all available models.
"""
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def find_all_models(results_dir: str = "results", seed: str = "seed_10"):
    """Find all model files in the results directory."""
    models = []
    base_path = Path(results_dir) / seed
    
    if not base_path.exists():
        logging.warning(f"Results directory not found: {base_path}")
        return models
    
    # Find all best_model.pt files
    for model_file in base_path.rglob("best_model.pt"):
        # Extract structure and target key from path
        # Path format: results/seed_10/R7-H2/E_H/best_model.pt
        parts = model_file.parts
        if len(parts) >= 4:
            structure = parts[-3]  # e.g., R7-H2
            target_key = parts[-2]  # e.g., E_H
            scaler_dir = model_file.parent
            
            models.append({
                "structure": structure,
                "target_key": target_key,
                "model_path": str(model_file),
                "scaler_dir": str(scaler_dir),
            })
    
    return models


def run_inference_measurement(model_info: dict, num_runs: int = 100):
    """Run inference time measurement for a single model."""
    structure = model_info["structure"]
    target_key = model_info["target_key"]
    model_path = model_info["model_path"]
    scaler_dir = model_info["scaler_dir"]
    
    logging.info(f"\n{'='*60}")
    logging.info(f"Measuring inference time for {structure} - {target_key}")
    logging.info(f"{'='*60}")
    
    # Run the measurement script
    # Use python3 explicitly to ensure we get the right interpreter
    python_cmd = "python3" if sys.executable.endswith("python3") else sys.executable
    cmd = [
        python_cmd,
        "ablation/measure_inference_time.py",
        "--model-path", model_path,
        "--scaler-dir", scaler_dir,
        "--target-key", target_key,
        "--num-runs", str(num_runs),
        "--num-warmup", "10",
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        
        if result.returncode == 0:
            logging.info(f"✓ Successfully measured {structure} - {target_key}")
            # Print output
            if result.stdout:
                logging.info(result.stdout)
            return True
        else:
            logging.error(f"✗ Failed to measure {structure} - {target_key}")
            if result.stderr:
                logging.error(result.stderr)
            return False
    except Exception as e:
        logging.error(f"✗ Error measuring {structure} - {target_key}: {e}")
        return False


def collect_all_results(results_dir: str = "results", seed: str = "seed_10"):
    """Collect all inference time results into a summary."""
    base_path = Path(results_dir) / seed
    all_results = {}
    
    for result_file in base_path.rglob("inference_time_results.json"):
        parts = result_file.parts
        if len(parts) >= 4:
            structure = parts[-3]
            target_key = parts[-2]
            
            try:
                with open(result_file) as f:
                    data = json.load(f)
                    if structure not in all_results:
                        all_results[structure] = {}
                    all_results[structure][target_key] = data
            except Exception as e:
                logging.warning(f"Failed to load {result_file}: {e}")
    
    return all_results


def print_summary(all_results: dict):
    """Print a summary table of all inference times."""
    logging.info("\n" + "="*80)
    logging.info("INFERENCE TIME SUMMARY")
    logging.info("="*80)
    
    # Get all structures and target keys
    structures = sorted(all_results.keys())
    target_keys = set()
    for struct_results in all_results.values():
        target_keys.update(struct_results.keys())
    target_keys = sorted(target_keys)
    
    # Print header
    header = f"{'Structure':<12} " + " ".join([f"{key:>10}" for key in target_keys])
    logging.info(header)
    logging.info("-" * len(header))
    
    # Print rows
    for structure in structures:
        row = f"{structure:<12} "
        for target_key in target_keys:
            if target_key in all_results[structure]:
                mean_ms = all_results[structure][target_key].get(
                    "single_batch", {}
                ).get("mean_ms", 0.0)
                row += f"{mean_ms:>10.4f} "
            else:
                row += f"{'N/A':>10} "
        logging.info(row)
    
    # Print detailed statistics
    logging.info("\n" + "="*80)
    logging.info("DETAILED STATISTICS")
    logging.info("="*80)
    
    for structure in structures:
        logging.info(f"\n{structure}:")
        for target_key in sorted(all_results[structure].keys()):
            data = all_results[structure][target_key]
            single_batch = data.get("single_batch", {})
            logging.info(f"  {target_key}:")
            logging.info(f"    Mean:     {single_batch.get('mean_ms', 0):.4f} ms")
            logging.info(f"    Std:      {single_batch.get('std_ms', 0):.4f} ms")
            logging.info(f"    Min:      {single_batch.get('min_ms', 0):.4f} ms")
            logging.info(f"    Max:      {single_batch.get('max_ms', 0):.4f} ms")
            logging.info(f"    Throughput: {single_batch.get('throughput_samples_per_sec', 0):.2f} samples/sec")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run inference time measurements for all available models."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Results directory (default: results)",
    )
    parser.add_argument(
        "--seed",
        type=str,
        default="seed_10",
        help="Seed directory (default: seed_10)",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=100,
        help="Number of inference runs per model (default: 100)",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print summary of existing results, don't run measurements",
    )
    args = parser.parse_args()
    
    # Find all models
    models = find_all_models(args.results_dir, args.seed)
    
    if not models:
        logging.error(f"No models found in {args.results_dir}/{args.seed}")
        return
    
    logging.info(f"Found {len(models)} models to measure")
    
    if not args.summary_only:
        # Run measurements
        success_count = 0
        for i, model_info in enumerate(models, 1):
            logging.info(f"\n[{i}/{len(models)}] Processing {model_info['structure']} - {model_info['target_key']}")
            if run_inference_measurement(model_info, args.num_runs):
                success_count += 1
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Completed: {success_count}/{len(models)} models measured successfully")
        logging.info(f"{'='*60}")
    
    # Collect and print summary
    all_results = collect_all_results(args.results_dir, args.seed)
    if all_results:
        print_summary(all_results)
        
        # Save summary to JSON
        summary_path = Path(args.results_dir) / args.seed / "inference_time_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logging.info(f"\nSummary saved to: {summary_path}")
    else:
        logging.warning("No inference time results found")


if __name__ == "__main__":
    main()

