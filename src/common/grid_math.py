# src/common/grid_math.py
"""
Deterministic 3D grid indexing utilities.

This module provides a reversible mapping between logical 3D grid
coordinates (i, j, k) and a single flat array index suitable for
NumPy-compatible storage.

----------------------------------------------------------------------
Indexing Model
----------------------------------------------------------------------

We work with a logical domain that includes ghost layers:

    i, j, k ∈ [ -1, N ]

NumPy does not allow negative indices in flattened addressing, because
-1 refers to the *last* element of the array. To avoid catastrophic
wrap‑around bugs in stencil/derivative operators, we shift all
coordinates by +1 before flattening.

Thus the forward mapping is:

    index = (i + 1)
            + nx_buf * ( (j + 1) + ny_buf * (k + 1) )

This is a *nested* layout:

    fastest axis:   i
    middle axis:    j
    slowest axis:   k

The inverse mapping must exactly reverse this nested structure.
The original implementation used the expanded form of the equation,
which does not match the nested forward mapping and therefore produced
incorrect (j, k) for nontrivial grids.

The corrected inverse mapping below is fully consistent and guarantees
round‑trip reversibility:

    (i, j, k) → index → (i, j, k)

----------------------------------------------------------------------

Functions
----------------------------------------------------------------------
"""

def get_flat_index(i: int, j: int, k: int, nx_buf: int, ny_buf: int) -> int:
    """
    Convert 3D coordinates (i, j, k) into a flat array index.

    All coordinates are shifted by +1 to avoid negative indices in the
    flattened NumPy layout.

    Parameters
    ----------
    i, j, k : int
        Logical grid coordinates in the range [ -1, N ].
    nx_buf : int
        Total size of the x‑dimension including ghost cells.
    ny_buf : int
        Total size of the y‑dimension including ghost cells.

    Returns
    -------
    int
        Flat array index.
    """
    return (i + 1) + nx_buf * ((j + 1) + ny_buf * (k + 1))


def get_coords_from_index(index: int, nx_buf: int, ny_buf: int) -> tuple[int, int, int]:
    """
    Inverse of get_flat_index: convert a flat index back to (i, j, k).

    This correctly reverses the nested structure:

        index = (i+1) + nx_buf * ( (j+1) + ny_buf * (k+1) )

    Parameters
    ----------
    index : int
        Flat array index.
    nx_buf : int
        Total size of the x‑dimension including ghost cells.
    ny_buf : int
        Total size of the y‑dimension including ghost cells.

    Returns
    -------
    (i, j, k) : tuple[int, int, int]
        Logical grid coordinates in the range [ -1, N ].
    """
    # First peel off the i‑dimension
    jk = index // nx_buf
    i  = index % nx_buf

    # Then peel off j and k from the nested jk structure
    j = jk % ny_buf
    k = jk // ny_buf

    # Undo the +1 shift
    return i - 1, j - 1, k - 1
