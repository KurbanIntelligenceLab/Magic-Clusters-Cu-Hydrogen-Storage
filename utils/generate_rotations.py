import math
import os
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Atomic radii (in Angstroms) for visualization
ATOMIC_RADII = {
    "H": 0.31,
    "Cu": 1.32,
}

# CPK-like color palette (RGB format for matplotlib)
ATOMIC_COLORS = {
    "H": (1.0, 1.0, 1.0),  # White
    "Cu": (0.8, 0.5, 0.2),  # Copper
}


def fibonacci_sphere_points(n: int) -> np.ndarray:
    """Generate n points on a sphere using the Fibonacci sphere method."""
    phi = (1 + math.sqrt(5)) / 2
    points = []

    for k in range(n):
        z = 1 - (2 * k + 1) / n
        r = math.sqrt(1 - z * z)
        phi_k = (2 * math.pi * k * phi ** (-1)) % (2 * math.pi)

        x = r * math.cos(phi_k)
        y = r * math.sin(phi_k)
        points.append([x, y, z])

    return np.array(points)


def rodrigues_rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Generate rotation matrix using Rodrigues' formula."""
    axis = axis / np.linalg.norm(axis)
    theta = np.radians(angle)

    # Skew-symmetric matrix
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )

    # Rodrigues' formula
    R = (
        np.eye(3) * np.cos(theta)
        + (1 - np.cos(theta)) * np.outer(axis, axis)
        + np.sin(theta) * K
    )

    return R


def read_xyz_file(file_path: str) -> Tuple[List[str], np.ndarray]:
    """Read XYZ file and return atom types and coordinates."""
    with open(file_path, "r") as f:
        lines = f.readlines()

    num_atoms = int(lines[0].strip())
    atoms = []
    coords = []

    for line in lines[2 : 2 + num_atoms]:
        parts = line.strip().split()
        atoms.append(parts[0])
        coords.append([float(x) for x in parts[1:4]])

    return atoms, np.array(coords)


def write_xyz_file(file_path: str, atoms: List[str], coords: np.ndarray):
    """Write XYZ file with atom types and coordinates."""
    with open(file_path, "w") as f:
        f.write(f"{len(atoms)}\n")
        f.write("Generated rotated structure\n")
        for atom, coord in zip(atoms, coords):
            f.write(f"{atom} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")


def create_atom_image(
    coords: np.ndarray, atoms: List[str], size: int = 128
) -> np.ndarray:
    """Create an image of the molecule using matplotlib for 3D to 2D projection."""
    # Create figure with white background
    plt.figure(figsize=(size / 100, size / 100), dpi=100)
    ax = plt.gca()
    ax.set_facecolor("white")

    # Set equal aspect ratio and remove axes
    ax.set_aspect("equal")
    ax.axis("off")

    # Get the 2D projection (xy-plane)
    coords_2d = coords[:, :2]

    # Calculate bounds with padding
    min_coords = np.min(coords_2d, axis=0)
    max_coords = np.max(coords_2d, axis=0)
    padding = 0.2 * (max_coords - min_coords)
    bounds = [
        min_coords[0] - padding[0],
        max_coords[0] + padding[0],
        min_coords[1] - padding[1],
        max_coords[1] + padding[1],
    ]

    # Plot atoms as circles
    for (x, y), atom in zip(coords_2d, atoms):
        radius = ATOMIC_RADII.get(atom, 0.5)
        color = ATOMIC_COLORS.get(atom, (0.5, 0.5, 0.5))
        circle = plt.Circle((x, y), radius, color=color, alpha=0.8)
        ax.add_patch(circle)

    # Set plot limits
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])

    # Convert plot to image
    plt.tight_layout(pad=0)
    plt.savefig("temp.png", bbox_inches="tight", pad_inches=0, dpi=100)
    plt.close()

    # Read the saved image and resize to desired size
    image = cv2.imread("temp.png")
    image = cv2.resize(image, (size, size))

    # Clean up temporary file
    os.remove("temp.png")

    return image


def process_xyz_file(xyz_path: str, output_dir: str, num_rotations: int = 5):
    """Process a single XYZ file to generate rotated versions and images."""
    # Create output directories
    xyz_output_dir = os.path.join(output_dir, "xyz_files")
    image_output_dir = os.path.join(output_dir, "images")
    os.makedirs(xyz_output_dir, exist_ok=True)
    os.makedirs(image_output_dir, exist_ok=True)

    # Read original XYZ file
    atoms, coords = read_xyz_file(xyz_path)

    # Calculate center of mass
    center_of_mass = np.mean(coords, axis=0)

    # Generate rotation axes using Fibonacci sphere
    rotation_axes = fibonacci_sphere_points(9)  # We'll use first 5

    # Process original structure
    base_name = os.path.splitext(os.path.basename(xyz_path))[0]

    # Save original XYZ with _rot0 suffix
    original_xyz_path = os.path.join(xyz_output_dir, f"{base_name}_rot0.xyz")
    write_xyz_file(original_xyz_path, atoms, coords)

    # Generate and save original image with _rot0 suffix
    original_image = create_atom_image(coords, atoms)
    cv2.imwrite(os.path.join(image_output_dir, f"{base_name}_rot0.png"), original_image)

    # Generate rotated versions
    for i in range(num_rotations):
        # Get rotation axis and create rotation matrix
        axis = rotation_axes[i]
        R = rodrigues_rotation_matrix(axis, 30)  # 30 degrees rotation

        # Apply rotation
        rotated_coords = np.dot(coords - center_of_mass, R.T) + center_of_mass

        # Save rotated XYZ
        rotated_xyz_path = os.path.join(xyz_output_dir, f"{base_name}_rot{i + 1}.xyz")
        write_xyz_file(rotated_xyz_path, atoms, rotated_coords)

        # Generate and save rotated image
        rotated_image = create_atom_image(rotated_coords, atoms)
        cv2.imwrite(
            os.path.join(image_output_dir, f"{base_name}_rot{i + 1}.png"), rotated_image
        )


def main():
    # Define input and output directories
    base_dir = "new_data"
    input_dirs = [
        os.path.join(base_dir, "initial-strcutures"),
        os.path.join(base_dir, "optimized-structures"),
        os.path.join(base_dir, "optimized-structures-with-H2"),
    ]

    # Process each directory
    for input_dir in input_dirs:
        if not os.path.exists(input_dir):
            print(f"Directory not found: {input_dir}")
            continue

        # Create output directory
        output_dir = os.path.join(base_dir, f"rotated_{os.path.basename(input_dir)}")
        os.makedirs(output_dir, exist_ok=True)

        # Process all XYZ files in the directory
        for xyz_file in os.listdir(input_dir):
            if xyz_file.endswith(".xyz"):
                xyz_path = os.path.join(input_dir, xyz_file)
                print(f"Processing {xyz_path}")
                process_xyz_file(xyz_path, output_dir)


if __name__ == "__main__":
    main()
