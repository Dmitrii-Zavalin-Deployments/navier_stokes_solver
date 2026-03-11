# tests/step2/test_step2_output_matches_schema.py

import numpy as np
from scipy.sparse import issparse
from tests.helpers.solver_step2_output_schema import EXPECTED_STEP2_SCHEMA

from src.step2.orchestrate_step2 import orchestrate_step2
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy


def test_step2_output_matches_schema():
    """
    Verifies that Step 2 takes a Step 1 state and correctly builds 
    sparse operators and PPE structures.
    """
    # 1. Build canonical input from Step 1 (Dependency Rule)
    # We use a small 4x4x4 grid for speed
    step1_state = make_step1_output_dummy(nx=4, ny=4, nz=4)

    # 2. Run real Step 2 Orchestrator
    # This is the call that was missing!
    state = orchestrate_step2(step1_state)

    # 3. Check object attribute existence
    for key in EXPECTED_STEP2_SCHEMA:
        assert hasattr(state, key), f"SolverState missing attribute: {key}"

    # 4. SCALE GUARD: Verify Operators are Sparse Matrices
    # This is the core requirement of Phase C, Rule 7
    assert isinstance(state.operators, dict)
    for op in ["laplacian", "divergence", "gradient"]:
        assert op in state.operators
        assert issparse(state.operators[op]), f"Operator {op} must be sparse!"

    # 5. PPE System Check
    assert isinstance(state.ppe, dict)
    assert issparse(state.ppe.get("A")), "PPE System matrix 'A' must be sparse!"
    assert isinstance(state.ppe.get("solver_type"), str)

    # 6. Mask Semantics (Refined in Step 2)
    assert isinstance(state.is_fluid, np.ndarray)
    assert isinstance(state.is_boundary_cell, np.ndarray)
    assert isinstance(state.is_solid, np.ndarray)

    # 7. Schema Validation via JSON-safe conversion
    # This ensures the 'to_json_safe' we fixed earlier works with real sparse data
    json_state = state.to_json_safe()
    
    # In JSON, sparse matrices should now be metadata dictionaries, not lists
    assert isinstance(json_state["operators"]["laplacian"], dict)
    assert "nnz" in json_state["operators"]["laplacian"]
    
    # Flags
    assert state.ready_for_time_loop is False  # Step 2 isn't the final step