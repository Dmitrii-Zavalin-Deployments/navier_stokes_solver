# tests/quality_gates/logic_gate/test_step3_mms.py

from src.common.field_schema import FI
from src.common.simulation_context import SimulationContext
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
    nx, ny, nz = 4, 4, 4
    # FIX: Pass dimensions to helper to ensure mask.data length == 64
    solver_input = create_validated_input(nx=nx, ny=ny, nz=nz)
    
    # Pathing Fix: SolverInput properties set directly
    solver_input.grid.nx = nx
    solver_input.grid.ny = ny
    solver_input.grid.nz = nz
    
    # Compliance Rule 4: Wrap in SimulationContext SSoT container
    context = SimulationContext(input_data=solver_input, config=None)
    
    # Setup Step 2 output state (The "Foundation" and "Wiring")
    state = make_step2_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # 2. Logic-Layer Traversal (Rule 1: Pointer Density Check)
    # STRATEGIC FIX: We select a CORE block (from state.stencil_matrix).
    # Since its center is NOT a ghost, it will NOT be short-circuited.
    block = state.stencil_matrix[0] 
    
    # Identify the ghost neighbor (i_minus) which is at index (-1, 0, 0) relative to core
    ghost_cell = block.i_minus
    assert ghost_cell.is_ghost is True, "MMS Setup Error: Traversal failed to find Ghost Cell."

    # 3. Setup: "Numerical Drift" Injection
    # We "poison" the ghost cell with 1.0. 
    # The Boundary Applier should reset this to 0.0 during orchestration.
    ghost_cell.set_field(FI.VX, 1.0) 
    
    # 4. Action: Execute Step 3 Orchestration
    # Because 'block' is a Core block, this will trigger Phase 1 (Predictor + Boundary Apply)
    updated_block, _ = orchestrate_step3(
        block=block,
        context=context,
        state_grid=state.grid,                 # Geometric Context (Rule 4)
        state_bc_manager=state.boundary_conditions, # Physical Context (Rule 4)
        is_first_pass=True
    )

    # 5. Verification: Boundary Enforcement (Rule 7: Atomic Truth)
    # The 'apply_boundary_values' inside the Core block orchestration 
    # must have sanitized the ghost neighbor's VX field.
    final_vx = ghost_cell.get_field(FI.VX)
    
    assert final_vx != 1.0, (
        "MMS FAILURE: Ghost short-circuit or Pointer integrity issue. "
        "The logic layer failed to mutate the NumPy Foundation in-place."
    )
    assert final_vx == 0.0, (
        f"MMS FAILURE: Boundary Enforcement mismatch. Got {final_vx}, expected 0.0 (No-Slip)."
    )

    # 6. Verification: Predictor Hydration (Rule 1: Field Precision Audit)
    # Ensure VX_STAR (intermediate field) was updated in the core cell.
    vx_star_val = block.center.get_field(FI.VX_STAR)
    
    assert vx_star_val != 0.0, (
        "MMS FAILURE: Predictor step (VX_STAR) failed to hydrate the Foundation buffer. "
        "Check physics kernel in src/step3/predictor.py."
    )

    # 7. SSoT Final Check (Rule 4)
    # Verify no illegal facade properties were created during orchestration.
    assert not hasattr(state, 'nx'), "Rule 4 Breach: Facade 'nx' detected on SolverState."