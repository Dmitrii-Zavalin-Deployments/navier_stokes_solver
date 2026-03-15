# src/common/grid_math.py

def get_flat_index(i: int, j: int, k: int, nx_buf: int, ny_buf: int) -> int:
    """
    Standardized index calculation: (i, j, k) -> 1D.
    
    Encapsulation: 
    We apply a +1 offset to all coordinates. This shifts the logical domain 
    [ -1, nx ] into the non-negative index space [ 0, nx+1 ] required for 
    NumPy-compatible flat array access. 
    
    Why: NumPy does not support negative indexing for multi-dimensional 
    array flattening; an index of -1 resolves to the end of the array, 
    triggering catastrophic wrap-around bugs during derivative calculations.
    """
    return (i + 1) + nx_buf * ((j + 1) + ny_buf * (k + 1))

def get_coords_from_index(index: int, nx_buf: int, ny_buf: int) -> tuple[int, int, int]:
    """
    Standardized inverse mapping: 1D -> (i, j, k).
    
    Encapsulation:
    Reverses the +1 shift applied by get_flat_index to return coordinates 
    to the logical domain space [ -1, nx ].
    """
    k = (index // (nx_buf * ny_buf)) - 1
    rem = index % (nx_buf * ny_buf)
    j = (rem // nx_buf) - 1
    i = (rem % nx_buf) - 1
    return i, j, k