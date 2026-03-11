# tests/step3/test_predict_velocity.py

import numpy as np
from scipy.sparse import eye

from src.step3.predict_velocity import predict_velocity
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


def test_predict_velocity_viscous_diffusion():
    """Test if velocity increases correctly via diffusion."""
    state = make_step2_output_dummy(nx=4, ny=4, nz=4)
    # Logic: 1.0 (initial) + dt/rho * mu * laplacian * U
    # 1.0 + 1.0/1.0 * 1.0 * 1.0 * 1.0 = 2.0
    state.constants["rho"] = 1.0
    state.constants["mu"] = 1.0 
    state.config["dt"] = 1.0
    
    u_size = state.fields["U"].size
    state.operators["lap_u"] = eye(u_size, format="csr")
    state.operators["advection_u"] = eye(u_size, format="csr") * 0.0
    state.fields["U"][:] = 1.0
    
    predict_velocity(state)
    
    internal_u_star = state.intermediate_fields["U_star"]
    assert np.allclose(internal_u_star, 2.0)