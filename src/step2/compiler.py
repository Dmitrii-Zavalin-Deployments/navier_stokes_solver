# src/step2/compiler.py

import numpy as np

from .cell import Cell


def GET_CELL_ATTRIBUTES():
    """
    Single Source of Truth for the cell matrix column order.
    To add a new physical or topological attribute, update the Cell class
    and append the property name here.
    """
    return ['x', 'y', 'z', 'vx', 'vy', 'vz', 'p', 'mask', 'is_ghost']

def cell_to_numpy_row(cell: Cell) -> np.ndarray:
    """
    Dynamically maps Cell properties to a NumPy row based on GET_CELL_ATTRIBUTES.
    
    This serialization process ensures that the order in the matrix always 
    matches the metadata, and strictly enforces the float64 format required 
    by downstream sparse matrix operations.
    """
    # Use getattr to pull values dynamically. 
    # We cast to float to ensure booleans (is_ghost) and ints (mask) 
    # are converted to the standard float64 matrix representation.
    return np.array([
        float(getattr(cell, attr)) for attr in GET_CELL_ATTRIBUTES()
    ], dtype=np.float64)