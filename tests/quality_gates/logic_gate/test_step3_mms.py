# tests/quality_gates/logic_gate/test_step3_mms.py

import pytest
from src.step3.orchestrate_step3 import orchestrate_step3
from src.common.field_schema import FI
from tests.helpers.solver_step2_output_dummy import solver_step2_output_dummy

def test_logic_gate_3_physics_boundary_sync():
    """
    Verification: apply_boundary_values resets ghost cell to physical boundary value (0.0).
    Target: src/step3/orchestrate_step3.py
    """
    # 1. Setup: Load Step 2 state and manually pollute a ghost cell
    state = solver_step2_output_dummy
    ghost_idx = 0 # Assume first index is a boundary ghost
    state.fields.data[ghost_idx, FI.VX] = 1.0 # Pollute with non-physical value

    # 2. Action
    # This triggers the predictor and the boundary applier dispatcher
    state_out, _ = orchestrate_step3(state, is_first_pass=True)

    # 3. Verification
    # Boundary applier (No-Slip) should have reset VX to 0.0
    final_val = state_out.fields.data[ghost_idx, FI.VX]
    assert final_val == 0.0, f"Boundary Enforcement Failed: Ghost VX is {final_val}, expected 0.0"