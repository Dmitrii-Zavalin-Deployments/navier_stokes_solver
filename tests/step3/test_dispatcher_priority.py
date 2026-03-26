# tests/step3/test_dispatcher_priority.py

import logging

import pytest

from src.common.solver_state import SolverState
from src.step2.factory import get_cell
from src.step3.boundaries.dispatcher import get_applicable_boundary_configs

# Rule 7: Setup Granular Traceability
logger = logging.getLogger(__name__)

@pytest.fixture
def real_state():
    """
    Rule 5: Provides a real, minimal SolverState for deterministic 
    initialization of Cells and Stencils.
    """
    # 10x10x10 Core Grid
    config = {
        "grid": {"nx": 10, "ny": 10, "nz": 10, "dx": 0.1, "dy": 0.1, "dz": 0.1},
        "fluid_properties": {"density": 1.0, "viscosity": 0.01},
        "simulation_parameters": {"time_step": 0.001},
        "initial_conditions": {"velocity": [0,0,0], "pressure": 0.0},
        "external_forces": {"force_vector": [0,0,0]}
    }
    state = SolverState(config)
    # Ensure all masks are fluid (1) by default
    state.mask.mask[:] = 1 
    return state

def create_real_test_block(i, j, k, state, mask=1):
    """
    Rule 9 Bridge: Uses the production Factory to create a real 
    StencilBlock. This verifies coordinate-logic (i, j, k) is correct.
    """
    from src.common.stencil_block import StencilBlock
    
    # Use Factory to get a real Cell (this sets index/nx_buf/ny_buf correctly)
    center = get_cell(i, j, k, state)
    center.mask = mask # Manually set mask for specific test scenarios
    
    # Define neighbors via Factory to maintain pointer integrity
    nb = {
        "i_minus": get_cell(i-1, j, k, state), "i_plus": get_cell(i+1, j, k, state),
        "j_minus": get_cell(i, j-1, k, state), "j_plus": get_cell(i, j+1, k, state),
        "k_minus": get_cell(i, j, k-1, state), "k_plus": get_cell(i, j, k+1, state)
    }

    return StencilBlock(
        center=center, **nb,
        dx=state.grid.dx, dy=state.grid.dy, dz=state.grid.dz,
        dt=state.simulation_parameters.time_step,
        rho=state.fluid_properties.density,
        mu=state.fluid_properties.viscosity,
        f_vals=tuple(state.external_forces.force_vector)
    )

def test_spatial_priority_over_mask_real_testing(real_state, caplog):
    """
    VERIFICATION: Ensure x_min (i=0) takes priority over mask=0 (Solid).
    Uses the real Factory to verify that coordinate detection works.
    """
    # Create block at i=0 (x_min) but force its mask to 0 (Solid)
    block = create_real_test_block(i=0, j=5, k=5, state=real_state, mask=0)
    
    boundary_cfg = [
        {'location': 'x_min', 'type': 'dirichlet', 'values': {'u': 1e10}}, 
        {'location': 'solid', 'type': 'no-slip', 'values': {'u': 0.0}}
    ]

    with caplog.at_level(logging.DEBUG):
        result = get_applicable_boundary_configs(block, boundary_cfg, real_state.grid, {"type": "INTERNAL"})

    # Assert Logic: Boundary location should be detected as 'x_min' via real Cell.i
    assert result[0]['location'] == 'x_min'
    assert result[0]['values']['u'] == 1e10
    assert f"DISPATCH [Spatial]: Block {block.id}" in caplog.text

def test_interior_fluid_real_testing(real_state, caplog):
    """Verify interior cell (i=5) is correctly identified without boundary tags."""
    block = create_real_test_block(i=5, j=5, k=5, state=real_state, mask=1)

    with caplog.at_level(logging.DEBUG):
        result = get_applicable_boundary_configs(block, [], real_state.grid, {"type": "INTERNAL"})

    assert result[0]['location'] == 'interior'
    assert "DISPATCH [Spatial]" not in caplog.text

def test_missing_config_detection_real_testing(real_state):
    """Rule 5: System must crash if i=0 is detected but no config exists."""
    block = create_real_test_block(i=0, j=5, k=5, state=real_state)
    
    with pytest.raises(KeyError, match="Missing boundary definition for x_min"):
        get_applicable_boundary_configs(block, [], real_state.grid, {"type": "INTERNAL"})