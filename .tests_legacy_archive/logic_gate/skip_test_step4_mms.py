# tests/logic_gate/test_step4_mms.py

import numpy as np

from src.step2.orchestrate_step2 import orchestrate_step2
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy


def test_logic_gate_4_linear_advection_transport():
    """
    LOGIC GATE 4: Linear Advection.
    """
    nx, ny, nz = 4, 4, 4
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # CONSTITUTIONAL ALIGNMENT: 
    # Advection stencils require non-zero flow to compute upwind/interpolation logic
    state.fields.U = np.ones_like(state.fields.U)
    state.fields.V = np.ones_like(state.fields.V)
    state.fields.W = np.ones_like(state.fields.W)
    
    state = orchestrate_step2(state)
    
    assert state.advection.weights is not None
    assert np.any(state.advection.weights != 0), "Advection weights are all zero."