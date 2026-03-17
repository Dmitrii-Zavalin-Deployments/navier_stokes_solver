# tests/step4/test_apply_bc_minimal_grid.py

from src.step4.apply_boundary_conditions import apply_boundary_conditions
from src.step4.initialize_extended_fields import initialize_extended_fields
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy


def test_apply_bc_minimal_grid():
    state = make_step3_output_dummy(nx=1, ny=1, nz=1)
    initialize_extended_fields(state)

    state.config["boundary_conditions"] = [
        {"type": "inlet", "variable": "u", "direction": "x", "side": "min", "value": 3.0}
    ]

    # Should not raise any index errors
    apply_boundary_conditions(state)

    assert state.U_ext[0, 0, 0] == 3.0
