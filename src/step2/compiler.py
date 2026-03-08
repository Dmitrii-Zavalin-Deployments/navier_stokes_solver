# src/step2/compiler.py

import numpy as np

from .cell import Cell


def GET_CELL_ATTRIBUTES():
    """Single Source of Truth for the cell matrix column order."""
    return ['x', 'y', 'z', 'vx', 'vy', 'vz', 'p', 'mask', 'is_ghost']

def cell_to_numpy_row(cell: Cell) -> np.ndarray:
    """
    Dynamically maps Cell properties to a numpy row based on GET_CELL_ATTRIBUTES.
    This ensures that the order in the matrix always matches the metadata.
    """
    # We use getattr to pull the values dynamically
    # This means if you add a property to Cell, you only update GET_CELL_ATTRIBUTES
    return np.array([
        getattr(cell, attr) for attr in GET_CELL_ATTRIBUTES()
    ], dtype=np.float64)