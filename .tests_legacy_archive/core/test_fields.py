# tests/core/test_fields.py

import numpy as np

from src.step3.core.fields import create_pressure_field, create_velocity_field


def test_field_initialization():
    nx, ny, nz = 10, 10, 10
    val = 0.0
    
    # Audit Field Creation
    v = create_velocity_field(nx, ny, nz, val)
    p = create_pressure_field(nx, ny, nz, val)
    
    # Proof: Check shapes and uniform initialization
    assert v.shape == (3, nx, ny, nz)
    assert p.shape == (nx, ny, nz)
    assert np.all(v == val)
    assert np.all(p == val)