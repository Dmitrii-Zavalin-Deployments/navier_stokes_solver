# tests/step3/test_dispatcher_priority.py

import logging
from unittest.mock import MagicMock

from src.step3.boundaries.dispatcher import get_applicable_boundary_configs


def test_spatial_priority_over_mask(caplog):
    """
    VERIFICATION: Ensure x_min (Inflow) takes priority even if mask is 0 (Solid).
    This confirms the fix for the 'Ghost 1e10' issue.
    """
    # Setup Mocks
    block = MagicMock()
    block.id = "TestBlock_001"
    block.center.i, block.center.j, block.center.k = 0, 5, 5 # On x_min face
    block.center.mask = 0                                   # Technically a solid cell
    
    grid = MagicMock()
    grid.nx, grid.ny, grid.nz = 10, 10, 10
    
    domain_cfg = {"type": "INTERNAL"}
    boundary_cfg = [
        {'location': 'x_min', 'values': {'u': 1e10}}, # The "Explosion"
        {'location': 'solid', 'values': {'u': 0.0}}
    ]

    # Execute with DEBUG logging enabled
    with caplog.at_level(logging.DEBUG):
        result = get_applicable_boundary_configs(block, boundary_cfg, grid, domain_cfg)

    # 1. Assert Logic Result: Should be the 1e10 inflow, NOT 0.0 solid
    assert result[0]['location'] == 'x_min'
    assert result[0]['values']['u'] == 1e10

    # 2. Assert Logger Evidence: Verify the spatial-first path was taken
    assert "DISPATCH [Spatial]: Block TestBlock_001 caught by x_min" in caplog.text
    # Ensure it DID NOT fall through to the mask logic
    assert "DISPATCH [Mask]" not in caplog.text

def test_interior_fallback(caplog):
    """Verify interior fluid doesn't trigger boundary logs."""
    block = MagicMock()
    block.id = "InteriorBlock"
    block.center.i, block.center.j, block.center.k = 5, 5, 5
    block.center.mask = 1
    
    grid = MagicMock()
    grid.nx, grid.ny, grid.nz = 10, 10, 10

    with caplog.at_level(logging.DEBUG):
        get_applicable_boundary_configs(block, [], grid, {"type": "INTERNAL"})

    assert "DISPATCH [Interior]" in caplog.text