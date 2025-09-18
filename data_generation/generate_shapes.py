"""
3D Voxel Snake Generator and Renderer

This module provides functionality to generate 3D voxel structures that resemble
snake-like configurations and render them as images from different viewpoints.
"""

import math
import os
import random
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from typing import List, Tuple, Set, Iterable, Optional
import glob

try:
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from tqdm import tqdm
    from PIL import Image
    HAS_DEPENDENCIES = True
except ImportError as e:
    print(f"Warning: Missing dependencies: {e}")
    HAS_DEPENDENCIES = False

# Constants
_NEIGH = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
DIRS: List[Tuple[int, int, int]] = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
]

# Image resolution constants
IMAGE_SIZE = (128, 128)  # Final image size (width, height)
FIGURE_SIZE = (10, 10)   # Matplotlib figure size for rendering

Voxel = Tuple[int, int, int]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _shift_to_origin(vox: List[Voxel]) -> List[Voxel]:
    """Shift voxel coordinates so the minimum is at origin."""
    if not vox:
        return []
    mins = [min(coord) for coord in zip(*vox)]
    dx, dy, dz = (-m for m in mins)
    return [(x + dx, y + dy, z + dz) for x, y, z in vox]


def orthogonals(d: Voxel) -> List[Voxel]:
    """Return the four directions that are perpendicular to `d`."""
    bad = {d, (-d[0], -d[1], -d[2])}
    return [v for v in DIRS if v not in bad]


def neighbour_count(v: Voxel, voxels: Set[Voxel]) -> int:
    """Number of occupied neighbours of voxel `v`."""
    x, y, z = v
    return sum((x + dx, y + dy, z + dz) in voxels for dx, dy, dz in DIRS)


def axis_of(d: Voxel) -> str:
    """Return 'x', 'y', or 'z' for a unit direction vector."""
    for idx, val in enumerate(d):
        if val != 0:
            return "xyz"[idx]
    raise ValueError("Invalid direction vector")


def flip_voxels(voxels: List[Voxel], axes: Tuple[str, ...] = ("x",)) -> List[Voxel]:
    """Flip voxels along specified axes."""
    if not voxels:
        return []

    xs, ys, zs = zip(*voxels)
    bounds = {
        "x": (min(xs), max(xs)),
        "y": (min(ys), max(ys)),
        "z": (min(zs), max(zs))
    }

    result = []
    for x, y, z in voxels:
        if "x" in axes:
            x = bounds["x"][1] - (x - bounds["x"][0])
        if "y" in axes:
            y = bounds["y"][1] - (y - bounds["y"][0])
        if "z" in axes:
            z = bounds["z"][1] - (z - bounds["z"][0])
        result.append((x, y, z))
    return result


def angle_between(e1_deg: float, a1_deg: float, e2_deg: float, a2_deg: float) -> float:
    """Calculate angle between two viewing directions in degrees."""
    e1, a1 = math.radians(e1_deg), math.radians(a1_deg)
    e2, a2 = math.radians(e2_deg), math.radians(a2_deg)

    v1 = (math.cos(e1) * math.cos(a1),
          math.cos(e1) * math.sin(a1),
          math.sin(e1))

    v2 = (math.cos(e2) * math.cos(a2),
          math.cos(e2) * math.sin(a2),
          math.sin(e2))

    dot = sum(p * q for p, q in zip(v1, v2))
    dot = max(-1.0, min(1.0, dot))  # clamp to handle numeric drift
    return math.degrees(math.acos(dot))


# ============================================================================
# VOXEL GENERATION
# ============================================================================

def generate_snake(
    N: int = 16,
    Lmin: int = 2,
    Lmax: int = 5,
    p_branch: float = 0.35,
    max_deg: int = 3,
    tries: int = 500,
    rng: Optional[random.Random] = None,
    is_2d: bool = False,
) -> List[Voxel]:
    """
    Create a 2-D or 3-D voxel snake.

    Parameters
    ----------
    N : int, default 16
        Number of voxels in the snake
    Lmin : int, default 2
        Minimum length of straight segments
    Lmax : int, default 5
        Maximum length of straight segments
    p_branch : float, default 0.35
        Probability of adding a branch at segment start
    max_deg : int, default 3
        Maximum number of occupied neighbors per voxel
    tries : int, default 500
        Maximum number of attempts to generate valid snake
    rng : Random, optional
        Random number generator instance
    is_2d : bool, default False
        If True, restricts generation to 2D (x-y plane)

    Returns
    -------
    list of (x, y, z) tuples, len(...) == N

    Raises
    ------
    RuntimeError if no admissible shape is found in `tries` attempts.
    """
    if rng is None:
        rng = random.Random()

    # Restrict available directions for 2D
    available_dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)] if is_2d else DIRS

    for _ in range(tries):
        voxels: Set[Voxel] = {(0, 0, 0)}
        order: List[Voxel] = [(0, 0, 0)]      # insertion order
        axes_used: Set[str] = set()                              # track spread

        # choose initial heading
        d = rng.choice(available_dirs)
        axes_used.add(axis_of(d))

        while len(voxels) < N:
            # ------------------ 1) grow main straight segment -----------------
            seg_len = min(rng.randint(Lmin, Lmax), N - len(voxels))
            x, y, z = order[-1]
            main_path: List[Voxel] = []

            # Try to trace the segment
            for _ in range(seg_len):
                x += d[0]; y += d[1]; z += d[2]
                nxt = (x, y, z)

                # overlap test
                if nxt in voxels:
                    break  # segment fails

                # degree tests (new voxel and its neighbours)
                if neighbour_count(nxt, voxels) >= max_deg:
                    break
                if any(
                    neighbour_count(nbr, voxels) + 1 > max_deg
                    for nbr in ((x + dx, y + dy, z + dz) for dx, dy, dz in available_dirs)
                    if nbr in voxels
                ):
                    break

                main_path.append(nxt)

            else:  # only executes if the for-loop did NOT break
                voxels.update(main_path)
                order.extend(main_path)
                axes_used.add(axis_of(d))

            # Abort and restart if main segment couldn't be placed
            if len(main_path) < seg_len:
                break

            if len(voxels) >= N:
                break

            # ------------------ 2) optional branch from segment start ----------
            if rng.random() < p_branch and len(voxels) < N and len(main_path) > 0:
                # Get the first voxel of the just-grown segment  
                seg_start_idx = len(order) - len(main_path) - 1
                if seg_start_idx >= 0:
                    sx, sy, sz = order[seg_start_idx]
                    # Filter orthogonals to only available directions
                    possible_branches = [dir for dir in orthogonals(d) if dir in available_dirs]
                    if possible_branches:  # Only proceed if we have valid branch directions
                        branch_dir = rng.choice(possible_branches)
                        bx, by, bz = sx + branch_dir[0], sy + branch_dir[1], sz + branch_dir[2]
                        br_vox = (bx, by, bz)
                        if (
                            br_vox not in voxels
                            and neighbour_count(br_vox, voxels) < max_deg
                            and neighbour_count((sx, sy, sz), voxels) + 1 <= max_deg
                        ):
                            voxels.add(br_vox)
                            order.append(br_vox)
                            axes_used.add(axis_of(branch_dir))

            # ------------------ 3) choose next heading -------------------------
            orths = [v for v in orthogonals(d) if v in available_dirs]
            rng.shuffle(orths)

            # Prefer directions on axes not visited yet
            unused_orths = [v for v in orths if axis_of(v) not in axes_used]
            for nd in unused_orths + orths:        # try unused first
                tx, ty, tz = order[-1]
                tx += nd[0]; ty += nd[1]; tz += nd[2]
                if (tx, ty, tz) not in voxels:
                    d = nd
                    axes_used.add(axis_of(d))
                    break
            else:
                break  # no viable turn -> restart outer attempt

        # ------------------ final acceptance test -----------------------------
        required_axes = {"x", "y"} if is_2d else {"x", "y", "z"}
        if len(voxels) == N and axes_used == required_axes:
            # Check if rotationally equivalent to its x-flip
            flipped = flip_voxels(order, axes=("x",))
            if not are_rotationally_equivalent(order, flipped, is_2d=is_2d):
                return order

    # Out of tries
    raise RuntimeError("Could not build a snake in the allotted attempts.")


# ============================================================================
# ROTATION AND EQUIVALENCE CHECKING
# ============================================================================

def _rotation_matrices():
    """Generate the 24 orientation-preserving 3×3 rotation matrices."""
    from itertools import permutations, product
    
    mats = []
    for perm in permutations(range(3)):
        inversions = sum(perm[i] > perm[j] for i in range(3) for j in range(i+1, 3))
        parity = inversions % 2

        for signs in product((1, -1), repeat=3):
            det = signs[0] * signs[1] * signs[2] * (-1)**parity
            if det == 1:  # rotation, not reflection
                mat = [[0]*3 for _ in range(3)]
                for row, (axis, s) in enumerate(zip(perm, signs)):
                    mat[row][axis] = s
                mats.append(tuple(tuple(r) for r in mat))
    return mats


ROTATIONS = _rotation_matrices()


def _apply_rotation(v: Voxel, mat) -> Voxel:
    """Apply rotation matrix to a voxel."""
    x, y, z = v
    return (
        mat[0][0]*x + mat[0][1]*y + mat[0][2]*z,
        mat[1][0]*x + mat[1][1]*y + mat[1][2]*z,
        mat[2][0]*x + mat[2][1]*y + mat[2][2]*z,
    )


def _canonicalize(voxels: Iterable[Voxel]) -> Set[Voxel]:
    """Translate voxels so the minimal coordinate is at origin."""
    voxels = list(voxels)
    if not voxels:
        return set()
    anchor = min(voxels)
    ax, ay, az = anchor
    return {(x-ax, y-ay, z-az) for x, y, z in voxels}


def _rotation_matrices_2d():
    """Generate the 4 rotation matrices for 2D rotations (90° increments)."""
    # 0°, 90°, 180°, 270° rotations in x-y plane
    return [
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),  # identity
        ((0, -1, 0), (1, 0, 0), (0, 0, 1)),  # 90°
        ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),  # 180°
        ((0, 1, 0), (-1, 0, 0), (0, 0, 1)),  # 270°
    ]


def are_rotationally_equivalent(A: Iterable[Voxel], B: Iterable[Voxel], is_2d: bool = False) -> bool:
    """
    Return True iff set A can be rotated (no reflection) and translated to match set B.
    
    Parameters
    ----------
    A : Iterable[Voxel]
        First set of voxels
    B : Iterable[Voxel]
        Second set of voxels
    is_2d : bool, default False
        If True, only check 2D rotations in x-y plane
    """
    A, B = list(A), list(B)
    if len(A) != len(B):
        return False
        
    B_canon = _canonicalize(B)
    rotations = _rotation_matrices_2d() if is_2d else ROTATIONS
    
    for mat in rotations:
        A_rot = [_apply_rotation(v, mat) for v in A]
        if _canonicalize(A_rot) == B_canon:
            return True
    return False


# ============================================================================
# RENDERING FUNCTIONS
# ============================================================================

def sample_view(
    *,
    elev_range: Tuple[Tuple[int, int], ...] = ((15, 75), (105, 165), (195, 255), (285, 345)),
    azim_sectors: Tuple[Tuple[int, int], ...] = ((15, 75), (105, 165), (195, 255), (285, 345))
) -> Tuple[int, int]:
    """Return a random (elev, azim) viewing angle."""
    elev_sector = random.choice(elev_range)
    elev = random.uniform(*elev_sector)
    azim_sector = random.choice(azim_sectors)
    azim = random.uniform(*azim_sector)
    return int(elev), int(azim)


def cube_vertices(origin: Voxel, size: float = 1) -> 'np.ndarray':
    """Generate vertices for a cube at given origin."""
    if not HAS_DEPENDENCIES:
        raise ImportError("NumPy is required for rendering")
    
    x, y, z = origin
    return np.array([
        [x, y, z], [x + size, y, z], [x + size, y + size, z], [x, y + size, z],
        [x, y, z + size], [x + size, y, z + size], [x + size, y + size, z + size], [x, y + size, z + size],
    ])


def cube_faces(verts: 'np.ndarray') -> List[List['np.ndarray']]:
    """Generate faces for a cube given its vertices."""
    return [
        [verts[j] for j in [0, 1, 2, 3]],  # bottom
        [verts[j] for j in [4, 5, 6, 7]],  # top
        [verts[j] for j in [0, 1, 5, 4]],  # front
        [verts[j] for j in [2, 3, 7, 6]],  # back
        [verts[j] for j in [1, 2, 6, 5]],  # right
        [verts[j] for j in [4, 7, 3, 0]],  # left
    ]


def _set_axes_equal(ax, pts: 'np.ndarray', padding: float = 0.1):
    """Force equal scaling on all axes with optional padding."""
    max_range = (pts.max(axis=0) - pts.min(axis=0)).max() / 2
    mid = pts.mean(axis=0)
    pad = max_range * padding
    
    for i, (set_lim, mid_val) in enumerate(zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], mid)):
        set_lim(mid_val - max_range - pad, mid_val + max_range + pad)


def plot_cubes(
    positions: List[Voxel],
    ax,
    *,
    size: float = 1,
    light_dir: Tuple[float, float, float] = (0.5, 0.6, 1.0),
    shade_base: float = 0.35,
    linewidth: float = 1.0, # linewidth: float = 0.2,
    edgecolor: str = "k",
    elev: float = 25,
    azim: float = 35,
    projection: str = "persp"
):
    """
    Draw one or many cubes in *ax* (an existing ``Axes3D``).

    Parameters
    ----------
    positions : sequence of (x, y, z)
        Lower-left-front corner of every cube.
    ax : matplotlib ``Axes3D``
        The subplot that will receive the geometry.
    size : float, default 1
        Edge length of each cube.
    light_dir : (3,) array-like, default (0.5, 0.6, 1.0)
        Direction of the head-light used for simple Lambert shading.
    shade_base : float, default 0.35
        Darkest shade (0 → black, 1 → white).
    linewidth, edgecolor
        Styling forwarded to ``Poly3DCollection``.
    elev : float, default 25
        Elevation viewing angle in degrees.
    azim : float, default 35
        Azimuth viewing angle in degrees.
    """
    if not HAS_DEPENDENCIES:
        raise ImportError("Matplotlib and NumPy are required for plotting")
    
    positions_array = np.atleast_2d(positions)

    # --- prepare shading -----------------------------------------------------
    light_dir_array = np.asarray(light_dir, dtype=float)
    light_dir_array /= np.linalg.norm(light_dir_array)

    faces, shades = [], []
    for pos in positions:
        verts = cube_vertices(pos, size)
        for face in cube_faces(verts):
            v1, v2 = np.subtract(face[1], face[0]), np.subtract(face[2], face[0])
            normal = np.cross(v1, v2).astype(np.float64)  # Convert to float64 before normalization
            normal /= np.linalg.norm(normal)
            intensity = np.clip(np.dot(normal, light_dir_array), 0.0, 1.0)
            #c = shade_base + (1.0 - shade_base) * intensity
            c = 0.75
            faces.append(face)
            shades.append((c, c, c))

    # --- actually draw -------------------------------------------------------
    coll = Poly3DCollection(
        faces, facecolors=shades, linewidths=linewidth, edgecolors=edgecolor
    )
    ax.add_collection3d(coll)

    # keep proportions & projection identical
    pts = np.concatenate([cube_vertices(p, size) for p in positions])
    _set_axes_equal(ax, pts)
    
    # Add padding around the shape
    padding = size * 0.1  # Adjust this multiplier to control padding amount
    max_range = (pts.max(axis=0) - pts.min(axis=0)).max() / 2
    mid = pts.mean(axis=0)
    ax.set_xlim(mid[0] - max_range - padding, mid[0] + max_range + padding)
    ax.set_ylim(mid[1] - max_range - padding, mid[1] + max_range + padding)
    ax.set_zlim(mid[2] - max_range - padding, mid[2] + max_range + padding)
    
    ax.set_proj_type(projection)
    ax.set_axis_off()
    ax.set_facecolor("white")
    # ax.set_facecolor("white")
    
    # set viewing angle
    ax.view_init(elev=elev, azim=azim)

    return coll  # the collection can be tweaked later if needed


# ============================================================================
# IMAGE PROCESSING
# ============================================================================

def process_voxel_views(
    args: Tuple[int, int, List[Voxel]], 
    *,
    output_dir: str,
    suffix: str,
    should_flip_second_view: bool,
    flip_axes: Tuple[str, ...],
    min_angle_diff: float,
    image_size: Tuple[int, int],
    figure_size: Tuple[int, int]
) -> None:
    """
    Process a voxel configuration into two different viewpoint images.
    
    Parameters
    ----------
    args : tuple
        Contains (i, j, voxel) where i, j are indices and voxel is the configuration
    output_dir : str
        Directory to save output images
    suffix : str
        Suffix to add to filename (e.g., 'S' for same, 'R' for reflected)
    should_flip_second_view : bool
        If True, flip the voxel configuration for the second view
    flip_axes : tuple of str
        Which axes to flip along (e.g., ('x',), ('y',), ('x', 'z'))
    min_angle_diff : float
        Minimum angle difference between the two views in degrees
    image_size : tuple of int
        Final image size (width, height)
    figure_size : tuple of int
        Matplotlib figure size for rendering
    """
    if not HAS_DEPENDENCIES:
        print("Skipping image processing - dependencies not available")
        return
        
    i, j, voxel = args
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    elev1, azim1 = sample_view()
    elev2, azim2 = sample_view()

    # Generate and save first view (always uses original voxel)
    fig, ax = plt.subplots(1, 1, figsize=figure_size, subplot_kw={'projection': '3d'})
    plot_cubes(voxel, ax, elev=elev1, azim=azim1)
    temp_path1 = os.path.join(output_dir, f"tmp_{i}_{j}_{elev1}_{azim1}_{suffix}.png")
    fig.savefig(temp_path1, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    final_path1 = os.path.join(output_dir, f"voxel_{i}_{j}_{elev1}_{azim1}_{suffix}.png")
    _process_and_save_image(temp_path1, final_path1, image_size)

    # For second view: use same initial 3D rotation but add 2D camera rotation
    second_voxel = flip_voxels(voxel, axes=flip_axes) if should_flip_second_view else voxel
    fig, ax = plt.subplots(1, 1, figsize=figure_size, subplot_kw={'projection': '3d'})
    plot_cubes(second_voxel, ax, elev=elev2, azim=azim2)
    temp_path2 = os.path.join(output_dir, f"tmp_{i}_{j}_{elev2}_{azim2}_{suffix}.png")
    fig.savefig(temp_path2, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    final_path2 = os.path.join(output_dir, f"voxel_{i}_{j}_{elev2}_{azim2}_{suffix}.png")
    _process_and_save_image(temp_path2, final_path2, image_size)


def process_voxel_pair(args: Tuple[int, int, List[Voxel]], *, base_output_dir: str) -> None:
    """Process a voxel configuration into two different viewpoint images (same voxel)."""
    process_voxel_views(
        args,
        output_dir=os.path.join(base_output_dir),
        suffix="S",
        should_flip_second_view=False,
        flip_axes=("x",),  # Not used but required
        min_angle_diff=20.0,
        image_size=IMAGE_SIZE,
        figure_size=FIGURE_SIZE
    )


def process_flipped_pair(args: Tuple[int, int, List[Voxel]], *, base_output_dir: str) -> None:
    """Process a voxel configuration and its flipped version into images."""
    process_voxel_views(
        args,
        output_dir=os.path.join(base_output_dir),
        suffix="R",
        should_flip_second_view=True,
        flip_axes=("x",),
        min_angle_diff=20.0,
        image_size=IMAGE_SIZE,
        figure_size=FIGURE_SIZE
    )


def _process_and_save_image(temp_path: str, final_path: str, image_size: Tuple[int, int] = IMAGE_SIZE) -> None:
    """Process a temporary image file and save the final version."""
    # Ensure output directory exists
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    
    image = Image.open(temp_path)
    image = image.crop((0, 0, image.height, image.height))
    image = image.resize(image_size)
    image = image.convert("L")
    image.save(final_path)
    os.remove(temp_path)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Generate voxel configurations and process them into images."""
    if not HAS_DEPENDENCIES:
        print("Cannot run main - missing required dependencies")
        return

    random.seed(42)
    np.random.seed(42)
    
    # Set processing parameters
    base_output_dir = "/"
    
    voxel_list = []
    
    max_attempts = 100000  # Prevent infinite loops
    attempts = 0

    pbar = tqdm(total=5000, desc="Generating voxel configurations")
    while len(voxel_list) < 5000 and attempts < max_attempts:
        attempts += 1
        try:
            voxels = generate_snake(
                N=np.random.randint(5, 10), 
                Lmin=2, 
                Lmax=4,
                p_branch=0.35, 
                max_deg=4, 
                tries=1000
            )

            # Skip if rotationally equivalent to its x-flip
            if are_rotationally_equivalent(voxels, flip_voxels(voxels, axes=("x",))):
                continue

            voxel_list.append(voxels)
            pbar.update(1)
                
        except RuntimeError:
            continue  # Try again if generation failed
    
    if attempts >= max_attempts:
        print(f"Warning: Reached maximum attempts ({max_attempts}). Only generated {len(voxel_list)} configurations.")

    print(f"Generated {len(voxel_list)} different voxel configurations")

    # Create argument lists for parallel processing
    args_list_tmp = [(i, j, voxel) for i, voxel in enumerate(voxel_list) for j in range(2)]
    

    from glob import glob
    # Get all voxel files sorted
    all_files = sorted(glob(os.path.join(base_output_dir, "voxel_*.png")))
    file_dict = defaultdict(list)
    for f in all_files:
        # Extract i,j from filename voxel_i_j_*.png
        basename = os.path.basename(f)
        i, j = map(int, basename.split('_')[1:3])
        file_dict[(i,j)].append(f)

    args_list = []
    for i, j, voxel in tqdm(args_list_tmp):
        files = file_dict[(i,j)]
        if len(files) == 4:
            continue
        else:
            # Delete any existing files
            for file in files:
                os.remove(file)
                print("Deleting", file)

            args_list.append((i, j, voxel))


    print(f"Processing {len(args_list)} pairs")

    # Process with multiprocessing - using the legacy functions for compatibility
    with Pool() as pool:
        list(tqdm(pool.imap(
            partial(process_voxel_pair, base_output_dir=base_output_dir),
            args_list
        ), total=len(args_list), desc="Processing regular pairs"))
        
        list(tqdm(pool.imap(
            partial(process_flipped_pair, base_output_dir=base_output_dir),
            args_list
        ), total=len(args_list), desc="Processing flipped pairs"))


if __name__ == "__main__":
    main()