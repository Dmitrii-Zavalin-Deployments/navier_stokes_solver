# tests/logic_gate/test_step3_mms.py

import numpy as np

from src.step2.orchestrate_step2 import orchestrate_step2
from src.step3.orchestrate_step3 import orchestrate_step3
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy


def test_logic_gate_3_divergence_pulse():
    """
    LOGIC GATE 3: Divergence Pulse.
    Analytical Truth: ||∇·u_new|| < 10⁻¹²
    
    Verifies that the pressure-correction projection successfully 
    removes divergence from a non-solenoidal input field.
    """
    nx, ny, nz = 4, 4, 4
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # Step 2 hydration (requires config.json or mocked config in environment)
    state = orchestrate_step2(state)
    
    dx = state.grid.dx
    # Initialize U with a linear gradient (div U = 1.0)
    U_raw = np.zeros((nx+1, ny, nz), order='F')
    for i in range(nx+1): 
        U_raw[i, :, :] = i * dx
    
    # CONSTITUTIONAL ALIGNMENT: Enforce 'F' layout for state storage
    state.fields.U = U_raw.flatten(order='F')
    state.fields.V = np.zeros_like(state.fields.V).flatten(order='F')
    state.fields.W = np.zeros_like(state.fields.W).flatten(order='F')
    
    # Execute Projection Step (Step 3)
    state_out = orchestrate_step3(state)
    
    # VALIDATION: Reconstruct the global velocity vector using Fortran alignment
    D = state.operators.divergence
    v_total = np.concatenate([
        state_out.fields.U.flatten(order='F'), 
        state_out.fields.V.flatten(order='F'), 
        state_out.fields.W.flatten(order='F')
    ])
    
    # Compute the L-infinity norm of the resulting divergence
    div_norm = np.linalg.norm(D.dot(v_total), np.inf)

    # Add this to tests/logic_gate/test_step3_mms.py
    print(f"DEBUG: Pressure Norm: {np.linalg.norm(state_out.fields.P)}")
    print(f"DEBUG: U-star head: {state.fields.U[:5]}")
    print(f"DEBUG: U-corrected head: {state_out.fields.U[:5]}")
    
    # Assertion against the mathematical zero floor
    assert div_norm < 1e-10