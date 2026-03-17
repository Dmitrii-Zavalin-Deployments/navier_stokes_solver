# tests/helpers/solver_step4_output_dummy.py

from src.common.field_schema import FI
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy


def make_step4_output_dummy(nx: int = 4, ny: int = 4, nz: int = 4, block_index: int = 0):
    """
    Returns a 'frozen' prototype representing the output of orchestrate_step4.
    
    Compliance:
    - Rule 4 (SSoT): Reflects schema-defined BC types (no-slip, inflow, etc.)
    - Logic: Values (u, v, w, p) are enforced across the targeted StencilBlock.
    """
    # Start from the Step 3 math results
    block = make_step3_output_dummy(nx=nx, ny=ny, nz=nz, block_index=block_index)
    
    # 1. Simulate Constraint Enforcement (Rule 9)
    # Based on your schema: values must include u, v, w, p
    # For a deterministic dummy, we set these to the 'target' verified state
    # This might affect the center cell OR neighbors depending on the BC location
    target_p = 0.01
    target_vel = 0.5
    
    # Enforce velocity and pressure fields as per Step 4 requirements
    block.center.set_field(FI.P, target_p)
    block.center.set_field(FI.VX, target_vel)
    block.center.set_field(FI.VY, target_vel)
    block.center.set_field(FI.VZ, target_vel)
    
    return block