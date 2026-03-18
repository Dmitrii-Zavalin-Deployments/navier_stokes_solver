# tests/helpers/solver_step3_output_dummy.py

"""
Archivist Testing: Snapshot-based Test Baseline (Step 3).

Compliance:
- Rule 4: SSoT (Hierarchy over convenience).
- Rule 9: Hybrid Memory Foundation (Fields initialized to verified state).
"""

from src.common.field_schema import FI
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


def make_step3_output_dummy(nx: int = 4, ny: int = 4, nz: int = 4, block_index: int = 0):
    """
    Returns a 'frozen' prototype representing the output of orchestrate_step3.
    
    This factory builds the full system state to ensure the Hybrid Memory 
    Foundation is structurally sound, then extracts the specific StencilBlock 
    required for structural parity audits.
    """
    # 1. Start from the Step 2 baseline (Full SolverState)
    state = make_step2_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # 2. Populate Foundation (Rule 9: Hybrid Memory Foundation)
    # Direct buffer manipulation ensures all StencilBlock views are updated.
    data = state.fields.data
    
    # Set Primary Velocities (divergence-free state post-correction)
    data[:, FI.VX] = 0.5
    data[:, FI.VY] = 0.5
    data[:, FI.VZ] = 0.5
    
    # Set Intermediate Velocities (Predictor results)
    data[:, FI.VX_STAR] = 0.51
    data[:, FI.VY_STAR] = 0.51
    data[:, FI.VZ_STAR] = 0.51
    
    # Set Pressure Fields (Post-synchronization)
    data[:, FI.P] = 0.01
    data[:, FI.P_NEXT] = 0.01

    # 3. Extraction
    # Returns the StencilBlock to match orchestrate_step3's return signature.
    return state.stencil_matrix[block_index]