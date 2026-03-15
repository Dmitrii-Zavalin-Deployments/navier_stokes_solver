# src/common/grid_math.py

def get_flat_index(i: int, j: int, k: int, nx_buf: int, ny_buf: int) -> int:
    """Standardized index calculation: (i, j, k) -> 1D."""
    # We use (i+1), (j+1), (k+1) to accommodate the 1-layer ghost border
    return (i + 1) + nx_buf * ((j + 1) + ny_buf * (k + 1))

def get_coords_from_index(index: int, nx_buf: int, ny_buf: int) -> tuple[int, int, int]:
    """Standardized inverse mapping: 1D -> (i, j, k)."""
    k = (index // (nx_buf * ny_buf)) - 1
    rem = index % (nx_buf * ny_buf)
    j = (rem // nx_buf) - 1
    i = (rem % nx_buf) - 1
    return i, j, k