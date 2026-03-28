# tests/quality_gates/logic_gate/test_step3_mms.py

from src.common.field_schema import FI
from src.common.simulation_context import SimulationContext
from src.step3.orchestrate_step3 import orchestrate_step3
from tests.helpers.solver_input_schema_dummy import create_validated_input
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
    Verification: Ensure the Boundary Applier mutates the cell it OWNS.
    Success Metric: A masked Core Block center is sanitized by the Applier.
    """
    # 1. Setup
    nx, ny, nz = 4, 4, 4
    state = make_step2_output_dummy(nx=nx, ny=ny, nz=nz)
    context = SimulationContext(input_data=create_validated_input(nx=nx, ny=ny, nz=nz), config=None)

    # 2. Target: Find a CORE block that represents a Boundary (Mask <= 0)
    # This block is an 'actor' in the matrix but requires boundary enforcement.
    try:
        target_block = next(b for b in state.stencil_matrix if b.center.mask <= 0)
    except StopIteration:
        # Fallback: If dummy has no walls, manually mask the first block for the test
        target_block = state.stencil_matrix[0]
        target_block.center.mask = -1 # Force Wall status

    # 3. Poison: Inject drift into the center cell
    target_block.center.set_field(FI.VX_STAR, 1.0)

    # 4. Action: Execute Orchestration
    # The dispatcher should see the mask and call the Boundary Applier.
    orchestrate_step3(
        block=target_block,
        context=context,
        state_grid=state.grid,
        state_bc_manager=state.boundary_conditions,
        is_first_pass=True
    )

    # 5. Verification: Ownership Audit
    # The owner MUST have sanitized its own VX_STAR to 0.0 (No-Slip).
    final_val = target_block.center.get_field(FI.VX_STAR)
    
    assert final_val != 1.0, (
        "MMS FAILURE: Boundary Applier ignored its own center cell. "
        f"Block {target_block.id} (Mask={target_block.center.mask}) remained poisoned."
    )
    assert final_val == 0.0, f"Expected 0.0 (No-Slip), got {final_val}"