import json
import os
from typing import Any, Dict, List, Optional, Tuple


def read_energies_file(file_path: str) -> Dict[str, Dict[str, float]]:
    """Read the energies.txt file and return a dictionary of structure names and their properties."""
    energies: Dict[str, Dict[str, float]] = {}
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Skip header lines
    start_idx = 0
    for i, line in enumerate(lines):
        if "HOMO" in line and "LUMO" in line:
            start_idx = i + 2
            break

    for line in lines[start_idx:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 5:
            name = parts[0]
            energies[name] = {
                "HOMO": float(parts[1]),
                "LUMO": float(parts[2]),
                "Eg": float(parts[3]),
                "Ef_t": float(parts[4]),  # total energy from energies.txt
            }
    return energies


def read_formation_energies_file(
    file_path: str,
) -> Tuple[Dict[str, Dict[str, float]], Optional[float]]:
    """Read the Formation_energy-EF.txt file and return a dictionary of structure base names and their properties, and the H2 energy."""
    formation_energies: Dict[str, Dict[str, float]] = {}
    h2_energy: Optional[float] = None
    with open(file_path, "r") as f:
        lines = f.readlines()
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 5:
            name = parts[0]
            formation_energies[name] = {
                "Cu-H2": float(parts[1]),
                "Cu": float(parts[2]),
                "H2": float(parts[3]),
                "Ef_f": float(parts[4]),  # formation energy
            }
            h2_energy = float(
                parts[3]
            )  # last H2 value (should be the same for all rows)
    return formation_energies, h2_energy


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


def main() -> None:
    """Main function to create a knowledge graph from structure and energy files."""
    base_dir = "new_data"
    energies = read_energies_file(os.path.join(base_dir, "energies.txt"))
    formation_energies, h2_energy = read_formation_energies_file(
        os.path.join(base_dir, "Formation_energy-EF.txt")
    )
    structure_dirs: List[Tuple[str, bool]] = [
        ("rotated_initial-strcutures", False),
        ("rotated_optimized-structures", True),
        ("rotated_optimized-structures-with-H2", True),
    ]
    nodes: List[Dict[str, Any]] = []
    all_structures: set = set()
    for dir_name, has_energy in structure_dirs:
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
                # Add physical/chemical properties
                if has_energy and base_name in energies:
                    node.update(energies[base_name])
                # Map to formation energy using base name (strip -H2 if present)
                base_for_formation = (
                    base_name.split("-H2")[0] if "-H2" in base_name else base_name
                )
                if base_for_formation in formation_energies:
                    node.update(formation_energies[base_for_formation])
                # Add all rotation file paths
                node["rotations"] = gather_rotations(xyz_dir, base_name)
                nodes.append(node)
    # Add H2 node
    if h2_energy is not None:
        nodes.append({"id": "H2", "Ef_t": h2_energy})
    # Save to JSON
    output_file = os.path.join(base_dir, "knowledge_graph.json")
    with open(output_file, "w") as f:
        json.dump(nodes, f, indent=2)
    print(f"Created knowledge graph with {len(nodes)} nodes.\nSaved to: {output_file}")


if __name__ == "__main__":
    main()
