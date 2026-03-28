# tests/quality_gates/logic_gate/test_step3_mms.py

import pytest
import numpy as np
from src.common.field_schema import FI
from src.common.simulation_context import SimulationContext
from src.step3.orchestrate_step3 import orchestrate_step3
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy

def test_logic_gate_3_physics_boundary_sync():
    """
    Logic Gate 3: Physics & Boundary Sync Verification
    Verification: Confirms 'Center-Write, Neighbor-Read' Architectural Integrity.
    Compliance: Rule 0 & Rule 4.
    """
    # 1. Setup
    nx, ny, nz = 4, 4, 4
    state = make_step2_output_dummy(nx=nx, ny=ny, nz=nz)
    # create_validated_input used for SSoT context
    from tests.helpers.input_helper import create_validated_input 
    solver_input = create_validated_input(nx=nx, ny=ny, nz=nz)
    context = SimulationContext(input_data=solver_input, config=None)

    # 2. Logic-Layer Traversal: Select a CORE block
    block = state.stencil_matrix[0] 
    ghost_neighbor = block.i_minus
    
    # 3. Setup: "Numerical Poisoning" of the Neighbor
    # We set the neighbor to 1.0. 
    ghost_neighbor.set_field(FI.VX_STAR, 1.0) 

    # 4. Action: Execute Step 3 Orchestration on the CORE block
    orchestrate_step3(
        block=block,
        context=context,
        state_grid=state.grid,
        state_bc_manager=state.boundary_conditions,
        is_first_pass=True
    )

    # 5. VERIFICATION: Neighbor Read-Only Audit
    # If this fails (if value != 1.0), the block mutated its neighbor.
    final_neighbor_vx = ghost_neighbor.get_field(FI.VX_STAR)
    
    assert final_neighbor_vx == 1.0, (
        "ARCHITECTURAL BREACH: Neighbor mutation detected! "
        "A block must treat neighbors as Read-Only to prevent race conditions. "
        f"Expected 1.0 (poisoned value), got {final_neighbor_vx}."
    )


def test_logic_gate_3_center_mutation_audit():
    """
    Verification: Ensure the block correctly mutates its OWN center cell.
    This proves the Boundary Applier is active for the primary ownership cell.
    """
    # 1. Setup
    nx, ny, nz = 4, 4, 4
    state = make_step2_output_dummy(nx=nx, ny=ny, nz=nz)
    from tests.helpers.input_helper import create_validated_input
    context = SimulationContext(input_data=create_validated_input(nx=nx, ny=ny, nz=nz), config=None)

    # 2. Target a block that IS a Ghost Block (to trigger Boundary Applier)
    # In Step 3, core blocks run 'predictor', ghost blocks run 'applier'.
    ghost_block = next(b for b in state.stencil_matrix if b.center.is_ghost)
    
    # 3. Poison the OWNED center cell
    ghost_block.center.set_field(FI.VX_STAR, 1.0)

    # 4. Action: Execute Orchestration on the owner
    orchestrate_step3(
        block=ghost_block,
        context=context,
        state_grid=state.grid,
        state_bc_manager=state.boundary_conditions,
        is_first_pass=True
    )

    # 5. VERIFICATION: Center Mutation Audit
    # The center MUST be sanitized because the block owns this memory.
    final_center_vx = ghost_block.center.get_field(FI.VX_STAR)
    
    assert final_center_vx != 1.0, (
        "MMS FAILURE: Boundary Applier failed to mutate its own center VX_STAR. "
        "The block center should have been sanitized via No-Slip."
    )
    assert final_center_vx == 0.0, f"Expected 0.0, got {final_center_vx}"