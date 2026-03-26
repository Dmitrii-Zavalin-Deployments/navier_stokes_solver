# tests/step3/test_dispatcher_priority.py

import numpy as np
import pytest
import logging

from src.common.field_schema import FI
from src.common.cell import Cell
from src.common.stencil_block import StencilBlock
from src.step3.boundaries.dispatcher import get_applicable_boundary_configs

# Rule 7: Setup Traceability
logger = logging.getLogger(__name__)

def create_dispatch_block(i, j, k, mask=1, nx=10):
    """
    Rule 5/9: Creates a real block at specific coordinates.
    i=0 is x_min, i=nx-1 is x_max.
    """
    # Create real buffer for 1 cell
    buffer = np.zeros((1, FI.num_fields()))
    
    # Instantiate real Cell (Rule 9: Logic for Wiring)
    center = Cell(
        index=0, 
        fields_buffer=buffer, 
        nx_buf=3, ny_buf=3, 
        is_ghost=False
    )
    # Manually set logical coordinates for dispatching
    center.i, center.j, center.k = i, j, k
    center.mask = mask
    
    # Satisfy StencilBlock constructor (Rule 5)
    # Neighbors aren't needed for dispatch logic, but must exist
    nb = [None] * 6 
    
    return StencilBlock(
        center=center,
        i_minus=nb[0], i_plus=nb[1],
        j_minus=nb[2], j_plus=nb[3],
        k_minus=nb[4], k_plus=nb[5],
        dx=0.1, dy=0.1, dz=0.1, dt=0.01,
        rho=1.0, mu=0.01, f_vals=(0,0,0)
    )

@pytest.fixture
def grid_meta():
    """Real Grid Metadata (Rule 4: SSoT)."""
    class Grid:
        nx, ny, nz = 10, 10, 10
    return Grid()

def test_spatial_priority_over_mask_real_logic(grid_meta, caplog):
    """
    VERIFICATION: Ensure x_min logic wins even if the mask is 'Solid' (0).
    This confirms the fix for the boundary-condition hierarchy.
    """
    # 1. Block at x_min (i=0) but masked as Solid (0)
    block = create_dispatch_block(i=0, j=5, k=5, mask=0)
    
    boundary_cfg = [
        {'location': 'x_min', 'type': 'dirichlet', 'values': {'u': 1e10}}, 
        {'location': 'solid', 'type': 'no-slip', 'values': {'u': 0.0}}
    ]

    # 2. Execute Dispatcher
    with caplog.at_level(logging.DEBUG):
        result = get_applicable_boundary_configs(block, boundary_cfg, grid_meta, {"type": "INTERNAL"})

    # 3. Assert: x_min must be the chosen identity
    assert result[0]['location'] == 'x_min'
    assert result[0]['values']['u'] == 1e10
    
    # 4. Rule 7: Verify Traceability
    assert f"DISPATCH [Spatial]: Block {block.id}" in caplog.text
    assert "matched specialized config for x_min" in caplog.text

def test_interior_fluid_no_dispatch_logs(grid_meta, caplog):
    """Verify that a middle-of-the-grid fluid cell falls through correctly."""
    # i=5, j=5, k=5 is strictly Interior
    block = create_dispatch_block(i=5, j=5, k=5, mask=1)

    with caplog.at_level(logging.DEBUG):
        result = get_applicable_boundary_configs(block, [], grid_meta, {"type": "INTERNAL"})

    assert result[0]['location'] == 'interior'
    # No boundary logs should exist for an interior cell
    assert "DISPATCH [Spatial]" not in caplog.text
    assert "DISPATCH [Mask]" not in caplog.text

def test_missing_config_crash_rule_5(grid_meta):
    """Rule 5: Crash immediately if a boundary is detected but not configured."""
    block = create_dispatch_block(i=0, j=5, k=5) # x_min
    
    # Passing empty list [] for boundary_cfg should trigger the KeyError
    with pytest.raises(KeyError, match="Missing boundary definition for x_min"):
        get_applicable_boundary_configs(block, [], grid_meta, {"type": "INTERNAL"})