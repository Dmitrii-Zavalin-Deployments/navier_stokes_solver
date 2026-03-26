# tests/step3/test_dispatcher_priority.py

import logging
from unittest.mock import MagicMock

import pytest

from src.step3.boundaries.dispatcher import get_applicable_boundary_configs


def test_spatial_priority_over_mask(caplog):
    """
    VERIFICATION: Ensure x_min (Inflow) takes priority even if mask is 0 (Solid).
    This confirms the fix for the 'Ghost 1e10' issue by hitting Spatial logic first.
    """
    # 1. Setup Mocks with hierarchical attribute access
    block = MagicMock()
    block.id = "TestBlock_001"
    # Ensure block.center has the required coordinates and mask
    block.center.i, block.center.j, block.center.k = 0, 5, 5 
    block.center.mask = 0                                   
    
    grid = MagicMock()
    grid.nx, grid.ny, grid.nz = 10, 10, 10
    
    domain_cfg = {"type": "INTERNAL"}
    boundary_cfg = [
        {'location': 'x_min', 'type': 'dirichlet', 'values': {'u': 1e10}}, 
        {'location': 'solid', 'type': 'no-slip', 'values': {'u': 0.0}}
    ]

    # 2. Execute with DEBUG logging
    with caplog.at_level(logging.DEBUG):
        result = get_applicable_boundary_configs(block, boundary_cfg, grid, domain_cfg)

    # 3. Assert Logic Result: Spatial (x_min) must win over Mask (solid)
    assert result[0]['location'] == 'x_min'
    assert result[0]['values']['u'] == 1e10

    # 4. Assert Logger Evidence: Match the actual f-string in dispatcher.py
    assert "DISPATCH [Spatial]: Block TestBlock_001 matched specialized config for x_min" in caplog.text
    # Verify we exited before hitting Mask logic
    assert "DISPATCH [Mask]" not in caplog.text

def test_interior_fallback(caplog):
    """Verify interior fluid triggers the correct return path without boundary logs."""
    block = MagicMock()
    block.id = "InteriorBlock"
    # Set coordinates to be away from all boundaries (nx=10, so 5 is safe)
    block.center.i, block.center.j, block.center.k = 5, 5, 5
    block.center.mask = 1 # Fluid
    
    grid = MagicMock()
    grid.nx, grid.ny, grid.nz = 10, 10, 10

    with caplog.at_level(logging.DEBUG):
        result = get_applicable_boundary_configs(block, [], grid, {"type": "INTERNAL"})

    assert result[0]['location'] == 'interior'
    # Check that it didn't match Spatial or Mask
    assert "DISPATCH [Spatial]" not in caplog.text
    assert "DISPATCH [Mask]" not in caplog.text

def test_missing_spatial_config_raises_error(caplog):
    """Rule 5: Ensure system crashes if a face is detected but config is missing."""
    block = MagicMock()
    block.center.i = 0 # x_min
    block.center.j, block.center.k = 5, 5
    
    grid = MagicMock()
    grid.nx = 10
    
    # Empty boundary_cfg while on a boundary should trigger KeyError
    with pytest.raises(KeyError, match="Missing boundary definition for x_min"):
        get_applicable_boundary_configs(block, [], grid, {"type": "INTERNAL"})