# tests/operators/test_forces.py

import numpy as np

from src.step3.ops.forces import get_body_forces


def test_get_body_forces_explicit():
    # Setup - No defaults allowed
    nx, ny, nz = 5, 5, 5
    fx, fy, fz = 0.0, -9.81, 0.0
    
    forces = get_body_forces(nx, ny, nz, fx, fy, fz)
    
    # Audit: Ensure explicit input matching
    assert forces.shape == (3, nx, ny, nz)
    assert np.all(forces[0] == 0.0)
    assert np.all(forces[1] == -9.81)
    assert np.all(forces[2] == 0.0)