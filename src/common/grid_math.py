# src/common/grid_math.py

def get_flat_index(i: int, j: int, k: int, nx: int, ny: int, offset: int = 0) -> int:
    """
    Computes a flat index from 3D coordinates. 
    Use offset=1 for buffered grids (ghost cells), offset=0 for interior grids.
    """
    # Stride is determined by the dimensions passed
    return (i + offset) + nx * (j + offset) + (nx * ny) * (k + offset)

def get_coords_from_index(index: int, nx: int, ny: int, offset: int = 0) -> tuple[int, int, int]:
    """
    SSoT Mapping: Converts flat index to (i, j, k).
    Use offset=1 for buffered grids (ghost cells), offset=0 for interior grids.
    """
    xy_plane = nx * ny
    
    k = (index // xy_plane) - offset
    rem = index % xy_plane
    j = (rem // nx) - offset
    i = (rem % nx) - offset
    
    return i, j, k