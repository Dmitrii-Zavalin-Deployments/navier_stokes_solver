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
    # FIX: Pass dimensions to helper to ensure mask.data length matches (nx*ny*nz)
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
    # Select a CORE block. This ensures orchestrate_step3 proceeds to Phase 1.
    block = state.stencil_matrix[0] 
    
    # Identify the ghost neighbor (i_minus) 
    ghost_cell = block.i_minus
    assert ghost_cell.is_ghost is True, "MMS Setup Error: Traversal failed to find Ghost Cell."

    # 3. Setup: "Numerical Drift" Injection
    # STRATEGIC ALIGNMENT: We poison VX_STAR because the Boundary Applier 
    # targets the trial fields in Step 3 to maintain Transactional Integrity.
    ghost_cell.set_field(FI.VX_STAR, 1.0) 
    
    # 4. Action: Execute Step 3 Orchestration (Predictor Pass)
    # This triggers apply_boundary_values(block, rule) for the core block's neighbors.
    updated_block, _ = orchestrate_step3(
        block=block,
        context=context,
        state_grid=state.grid,
        state_bc_manager=state.boundary_conditions,
        is_first_pass=True
    )

    # 5. Verification: Trial Enforcement (Rule 7: Atomic Truth)
    # The 'apply_boundary_values' logic must have reset the ghost VX_STAR.
    # Success Metric: 1.0 (Dirty Trial) -> 0.0 (Sanitized via No-Slip)
    final_vx_star = ghost_cell.get_field(FI.VX_STAR)
    
    assert final_vx_star != 1.0, (
        "MMS FAILURE: Boundary Applier failed to mutate VX_STAR. "
        "Check if Core Block neighbor traversal is hitting the Ghost Cell."
    )
    assert final_vx_star == 0.0, (
        f"MMS FAILURE: Boundary Enforcement mismatch in VX_STAR. "
        f"Got {final_vx_star}, expected 0.0 (No-Slip)."
    )

    # 6. Verification: Transaction Isolation (Rule 9)
    # Ensure the base VX field was NOT changed. 
    # In your logic, VX is only updated during the ElasticManager commit.
    base_vx = ghost_cell.get_field(FI.VX)
    assert base_vx == 0.0, (
        "RULE 9 BREACH: Base field mutated before ElasticManager commitment. "
        "The Applier should only target STAR fields during the trial loop."
    )

    # 7. Verification: Predictor Hydration (Rule 1: Field Precision Audit)
    # Ensure VX_STAR (intermediate field) was updated in the core cell center.
    vx_star_val = block.center.get_field(FI.VX_STAR)
    assert vx_star_val != 0.0, (
        "MMS FAILURE: Predictor step failed to hydrate the Core center VX_STAR."
    )

    # 8. SSoT Final Check (Rule 4)
    assert not hasattr(state, 'nx'), "Rule 4 Breach: Facade 'nx' detected on SolverState."