# tests/step3/test_step3_output_dummy_schema.py

import numpy as np
from tests.helpers.solver_step3_output_schema import EXPECTED_STEP3_SCHEMA

from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy


def test_step3_dummy_matches_schema():
    state = make_step3_output_dummy()

    # ------------------------------------------------------------
    # Top-level keys
    # ------------------------------------------------------------
    for key in EXPECTED_STEP3_SCHEMA:
        assert hasattr(state, key), f"Missing key: {key}"

    # ------------------------------------------------------------
    # Grid / Config / Constants
    # ------------------------------------------------------------
    assert isinstance(state.grid, dict)
    assert isinstance(state.config, dict)
    assert isinstance(state.constants, dict)

    # ------------------------------------------------------------
    # Mask semantics
    # ------------------------------------------------------------
    assert isinstance(state.mask, np.ndarray)
    assert isinstance(state.is_fluid, np.ndarray)
    assert isinstance(state.is_boundary_cell, np.ndarray)

    # ------------------------------------------------------------
    # Fields
    # ------------------------------------------------------------
    for f in ["U", "V", "W", "P"]:
        assert f in state.fields
        assert isinstance(state.fields[f], np.ndarray)

    # ------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------
    bc = state.boundary_conditions
    assert bc is None or callable(bc)

    # ------------------------------------------------------------
    # Operators
    # ------------------------------------------------------------
    assert isinstance(state.operators, dict)
    for op in ["divergence", "grad_x", "grad_y", "grad_z", "lap_u", "lap_v", "lap_w"]:
        assert op in state.operators
        assert callable(state.operators[op])

    # ------------------------------------------------------------
    # PPE metadata
    # ------------------------------------------------------------
    assert isinstance(state.ppe, dict)
    for key in ["solver_type", "tolerance", "max_iterations",
                "ppe_is_singular", "rhs_builder", "iterations", "converged"]:
        assert key in state.ppe

    # ------------------------------------------------------------
    # Health (Step 3)
    # ------------------------------------------------------------
    health_schema = EXPECTED_STEP3_SCHEMA["health"]
    for key, typ in health_schema.items():
        assert key in state.health
        assert isinstance(state.health[key], typ)

    # ------------------------------------------------------------
    # History (Step 3)
    # ------------------------------------------------------------
    hist_schema = EXPECTED_STEP3_SCHEMA["history"]
    for key, typ in hist_schema.items():
        assert key in state.history
        assert isinstance(state.history[key], typ)
