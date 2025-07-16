import csv
import json
import os
from typing import Any, Dict, List, Tuple


def gather_rotations(xyz_dir: str, base_name: str) -> List[Dict[str, str]]:
    """Gather all rotation file paths for a given base structure name."""
    rotations: List[Dict[str, str]] = []
    for i in range(6):
        xyz_path = os.path.join(xyz_dir, f"{base_name}_rot{i}.xyz")
        image_path = os.path.join(
            xyz_dir.replace("xyz_files", "images"), f"{base_name}_rot{i}.png"
        )
        if os.path.exists(xyz_path) and os.path.exists(image_path):
            rotations.append({"xyz_path": xyz_path, "image_path": image_path})
    return rotations


def read_ground_truth_csv(file_path: str) -> Dict[str, Dict[str, Any]]:
    data = {}
    with open(file_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            struct = row["Structure"].strip()

            def safe_float(val):
                val = val.strip()
                return float(val) if val not in ("", "--") else 0.0

            def safe_int(val):
                val = val.strip()
                return int(val) if val not in ("", "--") else 0

            data[struct] = {
                "E_H": safe_float(row.get("E_H", "")),
                "E_L": safe_float(row.get("E_L", "")),
                "E_g": safe_float(row.get("E_g", "")),
                "E_f": safe_float(row.get("E_f", "")),
                "E_T": safe_float(row.get("E_T", "")),
                "E_F": safe_float(row.get("E_F", "")),
                "d_Cu-H": safe_float(row.get("d_Cu-H", "")),
                "N_cu": safe_int(row.get("N_cu", "")),
                "N_h": safe_int(row.get("N_h", "")),
            }
    return data


def main() -> None:
    """Main function to create a knowledge graph from ground truth CSV only."""
    base_dir = "data"
    structure_dirs: List[Tuple[str, bool]] = [
        ("rotated_optimized-structures", True),
        ("rotated_optimized-structures-with-H2", True),
    ]
    nodes: List[Dict[str, Any]] = []
    all_structures: set = set()
    ground_truth = read_ground_truth_csv(os.path.join(base_dir, "labels.csv"))
    for dir_name, _ in structure_dirs:
        dir_path = os.path.join(base_dir, dir_name)
        xyz_dir = os.path.join(dir_path, "xyz_files")
        if not os.path.exists(xyz_dir):
            continue
        for file in os.listdir(xyz_dir):
            if file.endswith(".xyz") and "_rot" in file:
                base_name = (
                    file.replace("_rot0.xyz", "")
                    .replace("_rot1.xyz", "")
                    .replace("_rot2.xyz", "")
                    .replace("_rot3.xyz", "")
                    .replace("_rot4.xyz", "")
                    .replace("_rot5.xyz", "")
                )
                if base_name in all_structures:
                    continue  # avoid duplicates
                all_structures.add(base_name)
                node: Dict[str, Any] = {"id": base_name}
                node["rotations"] = gather_rotations(xyz_dir, base_name)
                if base_name in ground_truth:
                    node.update(ground_truth[base_name])
                nodes.append(node)
    # Add H2 node if present in CSV
    if "H2" in ground_truth:
        h2_node = {"id": "H2"}
        h2_node.update(ground_truth["H2"])
        h2_node["rotations"] = None
        nodes.append(h2_node)

    # Build edges
    edges = []
    id_to_node = {node["id"]: node for node in nodes}
    for node in nodes:
        node_id = node["id"]
        # Adsorption edge: Cu_x -> Cu_x-H2
        if not node_id.endswith("-H2") and node_id + "-H2" in id_to_node:
            edges.append(
                {"source": node_id, "target": node_id + "-H2", "type": "adsorption"}
            )
        # Size increment edge: Cu_x -> Cu_{x+1}
        if node.get("N_cu") is not None and node.get("N_h") == 0:
            next_id = f"R{node['N_cu'] + 1}"
            if next_id in id_to_node and id_to_node[next_id].get("N_h") == 0:
                edges.append(
                    {"source": node_id, "target": next_id, "type": "size_increment"}
                )
        # Size increment edge for H2: Cu_x-H2 -> Cu_{x+1}-H2
        if node.get("N_cu") is not None and node.get("N_h") == 2:
            next_id = f"R{node['N_cu'] + 1}-H2"
            if next_id in id_to_node and id_to_node[next_id].get("N_h") == 2:
                edges.append(
                    {"source": node_id, "target": next_id, "type": "size_increment"}
                )
        # Cross edge: Cu_x -> Cu_{x+1}-H2
        if node.get("N_cu") is not None and node.get("N_h") == 0:
            cross_id = f"R{node['N_cu'] + 1}-H2"
            if cross_id in id_to_node:
                edges.append({"source": node_id, "target": cross_id, "type": "cross"})
        # NEW: Add bidirectional desorption and size decrement edges
        # Desorption: Cu_x-H2 -> Cu_x
        if node_id.endswith("-H2"):
            base_id = node_id[:-3]
            if base_id in id_to_node:
                edges.append(
                    {"source": node_id, "target": base_id, "type": "desorption"}
                )
        # Size decrement: Cu_{x+1} -> Cu_x (same doping)
        if node.get("N_cu") is not None:
            if node.get("N_h") == 0:
                prev_id = f"R{node['N_cu'] - 1}"
                if prev_id in id_to_node and id_to_node[prev_id].get("N_h") == 0:
                    edges.append(
                        {"source": node_id, "target": prev_id, "type": "size_decrement"}
                    )
            if node.get("N_h") == 2:
                prev_id = f"R{node['N_cu'] - 1}-H2"
                if prev_id in id_to_node and id_to_node[prev_id].get("N_h") == 2:
                    edges.append(
                        {"source": node_id, "target": prev_id, "type": "size_decrement"}
                    )

    # Remove nodes with '-initial' in id
    nodes = [node for node in nodes if not node["id"].endswith("-initial")]
    valid_ids = {node["id"] for node in nodes}
    edges = [
        edge
        for edge in edges
        if edge["source"] in valid_ids and edge["target"] in valid_ids
    ]

    # Save as dict with nodes and edges
    output = {"nodes": nodes, "edges": edges}
    output_file = os.path.join(base_dir, "knowledge_graph.json")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(
        f"Created knowledge graph with {len(nodes)} nodes and {len(edges)} edges.\nSaved to: {output_file}"
    )


if __name__ == "__main__":
    main()
