# tests/common/test_stencil_block.py

import logging
from unittest.mock import MagicMock
import pytest

from src.common.stencil_block import StencilBlock

def test_dt_sync_logging(caplog):
    """
    VERIFICATION: Ensure StencilBlock logs the time-step synchronization event.
    This is critical for tracking Elasticity Manager interventions.
    """
    # 1. Setup Mock Cells for the 7-point stencil topology
    # Order: center, i_minus, i_plus, j_minus, j_plus, k_minus, k_plus
    mock_cells = [MagicMock() for _ in range(7)]
    
    # 2. Instantiate StencilBlock with required physics parameters
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
    # Check for the exact log format defined in the setter: {self._id} updated dt -> {value:.4e}
    assert f"SYNC [Physics]: {block_id} updated dt -> 1.0000e-06" in caplog.text

def test_dt_instability_guard(caplog):
    """VERIFICATION: Non-positive dt must log a CRASH and raise ValueError."""
    mock_cells = [MagicMock() for _ in range(7)]
    
    # Explicitly providing all 14 required arguments (7 cells + 7 physics params)
    block = StencilBlock(
        *mock_cells, 
        0.1, 0.1, 0.1, 0.01, 1.0, 0.01, (0, 0, 0)
    )

    with caplog.at_level(logging.ERROR):
        # The source uses "Numerical Instability" as the error message prefix
        with pytest.raises(ValueError, match="Numerical Instability"):
            block.dt = -0.005

    # Check for the specific log string triggered during the stability crash
    assert "STABILITY CRASH" in caplog.text
    assert f"rejected invalid dt=-0.005" in caplog.text

def test_dt_zero_instability_guard():
    """VERIFICATION: dt = 0 is physically invalid and must trigger the guard."""
    mock_cells = [MagicMock() for _ in range(7)]
    block = StencilBlock(*mock_cells, 0.1, 0.1, 0.1, 0.01, 1.0, 0.01, (0, 0, 0))
    
    with pytest.raises(ValueError, match="dt must be positive"):
        block.dt = 0.0