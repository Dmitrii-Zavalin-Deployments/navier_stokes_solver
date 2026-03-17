# tests/step3/test_predict_velocity_stability.py

import numpy as np
import pytest

from src.step3.predict_velocity import predict_velocity
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


def test_predict_velocity_numerical_stability():
    """Verify finite results and respect staggered grid boundaries."""
    nx, ny, nz = 4, 4, 4
    state = make_step2_output_dummy(nx=nx, ny=ny, nz=nz)
    state.constants.update({"rho": 1.0, "mu": 0.01})
    state.config["dt"] = 0.001
    
    u_init = np.ones((nx + 1, ny, nz))
    v_init = np.zeros((nx, ny + 1, nz))
    w_init = np.zeros((nx, ny, nz + 1))
    
    # FIX: Package initial fields in a dictionary for the 'fields' argument
    initial_fields = {"U": u_init, "V": v_init, "W": w_init}

    try:
        u_star, v_star, w_star = predict_velocity(state, fields=initial_fields)
    except Exception as e:
        pytest.fail(f"Prediction logic failed: {e}")

    assert np.all(np.isfinite(u_star))
    assert u_star.shape == (nx + 1, ny, nz)