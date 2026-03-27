# tests/quality_gates/logic_gate/test_step3_mms.py

from src.common.field_schema import FI
from src.step3.orchestrate_step3 import orchestrate_step3
from tests.helpers.solver_input_schema_dummy import create_validated_input
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


def test_logic_gate_3_physics_boundary_sync():
    """
    Logic Gate 3: Physics & Boundary Sync Verification
    
    Analytical Challenge: In-place Mutation (Hybrid Memory Pattern)
    Success Metric: P^n -> P^n+1 & No-Slip Enforcement
    Compliance: Rule 0 (Data for Logic, Arrays for Math)
    Compliance: Rule 4 (Hierarchy over Convenience)
    """
    
    # 1. Setup: Explicit Input to avoid "Silent Failure" (Rule 5)
    # FIX: Use the base constructor and set attributes via the SSoT path.
    nx, ny, nz = 4, 4, 4
    context = create_validated_input()
    
    # Pathing Fix: Ensure we traverse through .input_data to satisfy Rule 4
    context.input_data.grid.nx = nx
    context.input_data.grid.ny = ny
    context.input_data.grid.nz = nz
    
    # Setup Step 2 output state (The "Foundation" and "Wiring")
    # Using the same dimension lock for the dummy state generator
    state = make_step2_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # 2. Logic-Layer Traversal (Rule 1: Pointer Density Check)
    # Select a block adjacent to a boundary via the object-graph (StencilBlock)
    # We use the first block in the list container (Rule 0: No Object Arrays)
    block = state.stencil_matrix[0]
    
    # Identify a ghost neighbor using pointer-based traversal
    ghost_cell = block.i_minus
    assert ghost_cell.is_ghost is True, "MMS Setup Error: Traversal failed to find Ghost Cell."

    # 3. Setup: "Numerical Drift" Injection
    # Manually pollute the ghost cell buffer to test the Boundary Firewall.
    # Rule 1 check: We mutate the logic-object, which updates the underlying NumPy buffer.
    ghost_cell.set_field(FI.VX, 1.0) 
    
    # 4. Action: Execute Step 3 Orchestration
    # We pass the block and the SSoT containers (state.grid, state.boundary_conditions).
    # This must leverage the 'field_ref' pointers inside the Cell objects (Rule 0).
    updated_block, _ = orchestrate_step3(
        block=block,
        context=context,
        state_grid=state.grid,                 # Geometric Context (Rule 4)
        state_bc_manager=state.boundary_conditions, # Physical Context (Rule 4)
        is_first_pass=True
    )

    # 5. Verification: Boundary Enforcement (Rule 7: Atomic Truth)
    # The 'apply_boundary_values' logic must have reset the ghost VX.
    # Success Metric: 1.0 (Dirty) -> 0.0 (Sanitized via No-Slip)
    final_vx = ghost_cell.get_field(FI.VX)
    
    assert final_vx != 1.0, (
        "MMS FAILURE: Logic layer failed to mutate the NumPy Foundation in-place. "
        "Check field_ref pointer integrity (Rule 0)."
    )
    assert final_vx == 0.0, (
        f"MMS FAILURE: Boundary Enforcement mismatch. Got {final_vx}, expected 0.0 (No-Slip)."
    )

    # 6. Verification: Predictor Hydration (Rule 1: Field Precision Audit)
    # Ensure VX_STAR (intermediate field) was updated in the 'Sink' (NumPy buffer).
    # We check the center cell of the block.
    vx_star_val = block.center.get_field(FI.VX_STAR)
    
    assert vx_star_val != 0.0, (
        "MMS FAILURE: Predictor step (VX_STAR) failed to hydrate the Foundation buffer. "
        "Check vectorization/broadcasting in src/step3/physics_engine.py."
    )

    # 7. SSoT Final Check (Rule 4)
    # Verify no illegal facade properties were created during orchestration.
    assert not hasattr(state, 'nx'), "Rule 4 Breach: Facade 'nx' detected on SolverState."