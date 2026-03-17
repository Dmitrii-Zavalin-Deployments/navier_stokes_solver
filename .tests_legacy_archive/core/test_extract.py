# tests/core/test_extract.py

import numpy as np

from src.step3.core.extract import get_interior_field


def test_get_interior_field():
    nx, ny, nz = 5, 5, 5
    
    # Audit: Vector field extraction
    v = np.random.rand(3, nx, ny, nz)
    v_int = get_interior_field(v)
    assert v_int.shape == (3, 3, 3, 3)
    
    # Audit: Scalar field extraction
    p = np.random.rand(nx, ny, nz)
    p_int = get_interior_field(p)
    assert p_int.shape == (3, 3, 3)