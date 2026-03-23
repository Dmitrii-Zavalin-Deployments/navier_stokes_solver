# tests/common/test_stencil_block.py

import pytest
import logging
from unittest.mock import MagicMock
from src.common.stencil_block import StencilBlock

def test_dt_sync_logging(caplog):
    """
    VERIFICATION: Ensure StencilBlock logs the time-step synchronization event.
    This is critical for tracking Elasticity Manager interventions.
    """
    # 1. Setup Mock Cells for the 7-point stencil
    mock_cells = [MagicMock() for _ in range(7)]
    
    # 2. Instantiate StencilBlock
    block = StencilBlock(
        *mock_cells, 
        dx=0.1, dy=0.1, dz=0.1, 
        dt=0.01, rho=1.0, mu=0.01, f_vals=(0, 0, 0)
    )
    
    block_id = block.id

    # 3. Simulate an Elasticity Downshift (Recovery Phase)
    new_dt = 1e-6
    with caplog.at_level(logging.DEBUG):
        block.dt = new_dt

    # --- ASSERTIONS ---
    assert block.dt == new_dt
    assert f"SYNC [Physics]: {block_id} updated dt -> 1.0000e-06" in caplog.text

def test_dt_instability_guard(caplog):
    """VERIFICATION: Negative dt must log an error and raise ValueError."""
    mock_cells = [MagicMock() for _ in range(7)]
    block = StencilBlock(
        *mock_cells, 0.1, 0.1, 0.1, 0.01, 1.0, 0.01, (0, 0, 0)
    )

    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValueError, match="Numerical Instability"):
            block.dt = -0.005

    assert "STABILITY CRASH" in caplog.text
    assert "rejected invalid dt=-0.005" in caplog.text