# tests/step3/test_step3_shape_integrity.py

import numpy as np
import pytest
from scipy.sparse import csr_matrix, eye

from src.step3.apply_boundary_conditions_post import apply_boundary_conditions_post
from src.step3.build_ppe_rhs import build_ppe_rhs
from src.step3.correct_velocity import correct_velocity
from src.step3.solve_pressure import solve_pressure
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


def test_step3_pipeline_shape_integrity():
    """
    Unit test to detect naming and shape mismatches across the Step 3 pipeline.
    This ensures the staggered grid (MAC grid) dimensions are maintained 
    from RHS calculation through to the final BC enforcement.
    """
    # 1. Setup State (Grid: 4x4x4)
    nx, ny, nz = 4, 4, 4
    state = make_step2_output_dummy(nx=nx, ny=ny, nz=nz)
    state.constants["rho"] = 1.0
    state.config["dt"] = 0.1
    
    # --- OPERATOR INJECTION (Fixes KeyError: 'grad_x') ---
    u_size = (nx + 1) * ny * nz
    v_size = nx * (ny + 1) * nz
    w_size = nx * ny * (nz + 1)
    p_size = nx * ny * nz
    vel_total_size = u_size + v_size + w_size

    # Inject mock divergence and PPE matrix
    state.operators["divergence"] = csr_matrix((p_size, vel_total_size))
    state.ppe["A"] = eye(p_size, format="csr") * -6.0

    # Inject mock gradient operators (Required by correct_velocity)
    state.operators["grad_x"] = lambda p: np.zeros((nx + 1, ny, nz))
    state.operators["grad_y"] = lambda p: np.zeros((nx, ny + 1, nz))
    state.operators["grad_z"] = lambda p: np.zeros((nx, ny, nz + 1))
    # -----------------------------------------------------

    # 2. Check Input Integrity (Initial MAC Grid setup)
    inputs = {
        "U": state.fields["U"], 
        "V": state.fields["V"], 
        "W": state.fields["W"], 
        "is_fluid": state.is_fluid
    }
    errors = validate_staggered_shapes(state, inputs)
    assert not errors, f"Initial state has shape errors: {errors}"

    # 3. Check RHS Output (Should be cell-centered)
    # Expected: (4, 4, 4)
    rhs = build_ppe_rhs(state, state.fields["U"], state.fields["V"], state.fields["W"])
    errors = validate_staggered_shapes(state, {"rhs_ppe": rhs})
    assert not errors, f"build_ppe_rhs output is invalid: {errors}"

    # 4. Check Pressure Solve Output (Should be cell-centered)
    # Expected: (4, 4, 4)
    P_new, _ = solve_pressure(state, rhs)
    errors = validate_staggered_shapes(state, {"P": P_new})
    assert not errors, f"solve_pressure output is invalid: {errors}"

    # 5. Check Velocity Correction (Should return to staggered shapes)
    # Expected: U(5,4,4), V(4,5,4), W(4,4,5)
    U_c, V_c, W_c = correct_velocity(state, state.fields["U"], state.fields["V"], state.fields["W"], P_new)
    errors = validate_staggered_shapes(state, {"U": U_c, "V": V_c, "W": W_c})
    assert not errors, f"correct_velocity output is invalid: {errors}"

    # 6. Check Post-BC Enforcement (Dictionary check)
    final_fields = apply_boundary_conditions_post(state, U_c, V_c, W_c, P_new)
    errors = validate_staggered_shapes(state, final_fields)
    
    if errors:
        pytest.fail(f"Shape/Naming Mismatches found in Final BC step:\n" + "\n".join(errors))

# --- INTERNAL HELPER FUNCTION ---

def validate_staggered_shapes(state, fields_to_check):
    """
    Diagnostic tool to verify MAC Grid shape consistency.
    Logic:
    - U is staggered in X (nx+1)
    - V is staggered in Y (ny+1)
    - W is staggered in Z (nz+1)
    - P, Masks, and RHS are cell-centered (nx, ny, nz)
    """
    nx, ny, nz = state.grid["nx"], state.grid["ny"], state.grid["nz"]
    
    expected = {
        "U": (nx + 1, ny, nz),
        "V": (nx, ny + 1, nz),
        "W": (nx, ny, nz + 1),
        "P": (nx, ny, nz),
        "is_fluid": (nx, ny, nz),
        "rhs_ppe": (nx, ny, nz)
    }
    
    mismatches = []
    
    for name, actual_field in fields_to_check.items():
        if name in expected:
            actual_shape = getattr(actual_field, "shape", None)
            if actual_shape != expected[name]:
                mismatches.append(
                    f"MISMATCH in '{name}': Expected {expected[name]}, got {actual_shape}"
                )
    
    return mismatches