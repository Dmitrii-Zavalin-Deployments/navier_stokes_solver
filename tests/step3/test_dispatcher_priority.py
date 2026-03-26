# tests/step3/test_dispatcher_priority.py

import pytest
import logging
import numpy as np

# Rule 4 & 5: Use the centralized dummy factory for SSoT
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy
from src.step2.factory import get_cell
from src.step3.boundaries.dispatcher import get_applicable_boundary_configs
from src.common.stencil_block import StencilBlock

logger = logging.getLogger(__name__)

@pytest.fixture
def real_state():
    """
    Rule 5: Deterministic Initialization using the centralized 
    Archivist Testing helper.
    """
    # Initialize a 10x10x10 grid via the official dummy helper
    return make_step1_output_dummy(nx=10, ny=10, nz=10)

def create_real_test_block(i, j, k, state, mask=1):
    """
    Rule 9 Bridge: Uses production Factory to wire a StencilBlock 
    based on the real hydrated SolverState.
    """
    # Ensure the mask buffer matches the test intent
    # Note: dummy creates mask as np.ones, we update it here
    state.mask.mask[i, j, k] = mask
    
    center = get_cell(i, j, k, state)
    
    # Neighborhood assembly via real Factory logic
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

# --- Logic Tests ---

def test_spatial_priority_over_mask_real_testing(real_state, caplog):
    """
    VERIFICATION: i=0 (x_min) must take priority even if mask=0.
    """
    # Setup: Spatial Boundary (i=0) + Solid Mask (0)
    block = create_real_test_block(i=0, j=5, k=5, state=real_state, mask=0)
    
    # Config defines both spatial and mask-based boundaries
    boundary_cfg = [
        {'location': 'x_min', 'type': 'dirichlet', 'values': {'u': 1e10}}, 
        {'location': 'solid', 'type': 'no-slip', 'values': {'u': 0.0}}
    ]

    with caplog.at_level(logging.DEBUG):
        # We pass real_state.grid and real_state.domain_configuration.to_dict() 
        # to match the dispatcher's signature
        domain_cfg = {"type": "INTERNAL"} 
        result = get_applicable_boundary_configs(block, boundary_cfg, real_state.grid, domain_cfg)

    # Spatial check (i=0) should have triggered 'x_min'
    assert result[0]['location'] == 'x_min'
    assert result[0]['values']['u'] == 1e10
    assert "DISPATCH [Spatial]" in caplog.text

def test_interior_fluid_real_testing(real_state, caplog):
    """Verify interior cells don't trigger boundary dispatch."""
    block = create_real_test_block(i=5, j=5, k=5, state=real_state, mask=1)

    with caplog.at_level(logging.DEBUG):
        result = get_applicable_boundary_configs(block, [], real_state.grid, {"type": "INTERNAL"})

    assert result[0]['location'] == 'interior'
    assert "DISPATCH [Spatial]" not in caplog.text

def test_missing_config_detection_real_testing(real_state):
    """Rule 5: Fail-fast if a detected boundary has no config."""
    block = create_real_test_block(i=0, j=5, k=5, state=real_state)
    
    with pytest.raises(KeyError, match="Missing boundary definition for x_min"):
        # Passing empty list for boundary_cfg should trigger the error for i=0
        get_applicable_boundary_configs(block, [], real_state.grid, {"type": "INTERNAL"})