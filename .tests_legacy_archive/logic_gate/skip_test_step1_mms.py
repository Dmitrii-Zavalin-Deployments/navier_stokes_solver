# tests/logic_gate/test_step1_mms.py

from src.solver_input import SolverInput
from src.step1.orchestrate_step1 import orchestrate_step1
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy


def test_logic_gate_1_structural_unit_cube():
    """
    LOGIC GATE 1: Structural Verification.
    Scenario: nx=2, ny=2, nz=2, L=1.0
    Analytical Truth: dx=0.5, dy=0.5, dz=0.5
    Verification: U=12, V=12, W=12, P=8 nodes.
    """
    # 1. Input: Unit Cube with 2x2x2 resolution
    input_data = solver_input_schema_dummy()
    input_data["grid"].update({
        "nx": 2, "ny": 2, "nz": 2,
        "x_min": 0.0, "x_max": 1.0,
        "y_min": 0.0, "y_max": 1.0,
        "z_min": 0.0, "z_max": 1.0
    })
    
    # 2. Execution
    state = orchestrate_step1(SolverInput.from_dict(input_data), iteration=0, time=0.0)
    
    # 3. Verification of Analytical Truth
    assert state.grid.dx == 0.5
    assert state.grid.dy == 0.5
    assert state.grid.dz == 0.5
    
    # Staggered Node Counts: (N+1)*N*N for velocities, N*N*N for pressure
    u_nodes = state.fields["U"].size
    v_nodes = state.fields["V"].size
    w_nodes = state.fields["W"].size
    p_nodes = state.fields["P"].size
    
    assert u_nodes == 12, f"Logic Gate 1 Failure: U_Vector expected 12, got {u_nodes}"
    assert v_nodes == 12
    assert w_nodes == 12
    assert p_nodes == 8