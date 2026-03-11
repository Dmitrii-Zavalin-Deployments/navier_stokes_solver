# tests/step4/test_step4_integration_roundtrip.py

from src.solver_state import SolverState
from src.step4.apply_boundary_conditions import apply_boundary_conditions
from src.step4.assemble_diagnostics import assemble_diagnostics
from src.step4.initialize_extended_fields import initialize_extended_fields
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy


def test_step4_integration_roundtrip():
    # Step 3 dummy
    state = make_step3_output_dummy(nx=3, ny=3, nz=3)

    # Simple BCs
    state.config["boundary_conditions"] = [
        {"location": "x_min", "type": "inflow", "values": {"u": 5.0}},
        {"location": "x_max", "type": "outflow"},
        {"location": "y_min", "type": "pressure", "values": {"p": 2.0}},
    ]

    # Full Step 4 pipeline
    initialize_extended_fields(state)
    apply_boundary_conditions(state)
    assemble_diagnostics(state)

    # Basic structural checks
    assert isinstance(state, SolverState)
    assert state.P_ext is not None
    assert state.U_ext is not None
    assert state.V_ext is not None
    assert state.W_ext is not None

    # BCs applied
    assert (state.U_ext[0, :, :] == 5.0).all()     # inflow
    assert (state.P_ext[:, 0, :] == 2.0).all()     # pressure Dirichlet

    # Diagnostics exist
    assert isinstance(state.step4_diagnostics, dict)
    assert "post_bc_max_velocity" in state.step4_diagnostics
