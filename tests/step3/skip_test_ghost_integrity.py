# tests/step3/test_ghost_integrity.py

import math

import numpy as np
import pytest

from src.common.field_schema import FI
from src.step3.ops.ghost_handler import sync_ghost_trial_buffers


class MockCell:
    """A Rule 9 compliant mock for Foundation-Object mapping."""
    __slots__ = ['index', 'fields_buffer', 'is_ghost']
    
    def __init__(self, index, fields_buffer, is_ghost=False):
        self.index = index
        self.fields_buffer = fields_buffer
        self.is_ghost = is_ghost

    def get_field(self, field_id: FI) -> float:
        return self.fields_buffer[self.index, field_id]

    def set_field(self, field_id: FI, value: float) -> None:
        self.fields_buffer[self.index, field_id] = value

@pytest.fixture
def ghost_stencil():
    """Sets up a StencilBlock centered on a Ghost cell with poisoned trial buffers."""
    # Create a small buffer for 1 cell
    buffer = np.zeros((1, FI.num_fields()))
    
    # 1. Initialize Foundation (Valid State)
    buffer[0, FI.VX] = 1.23
    buffer[0, FI.VY] = 4.56
    buffer[0, FI.VZ] = 7.89
    buffer[0, FI.P]  = 101.325
    
    # 2. Poison the Trial Buffers (Simulating a failed dt=0.8 run)
    buffer[0, FI.VX_STAR] = np.nan
    buffer[0, FI.VY_STAR] = np.inf
    buffer[0, FI.VZ_STAR] = -np.inf
    buffer[0, FI.P_NEXT]  = np.nan
    
    center_cell = MockCell(index=0, fields_buffer=buffer, is_ghost=True)
    
    # Create a minimal block (neighbors aren't needed for this atomic test)
    class MinimalBlock:
        def __init__(self, center):
            self.center = center
            
    return MinimalBlock(center=center_cell)

def test_atomic_ghost_nan_recovery(ghost_stencil):
    """
    Rule 7 Verification: Ensures poisoned ghost buffers are reset to Foundation.
    """
    block = ghost_stencil
    
    # Pre-condition check: verify the 'poison' exists
    assert not math.isfinite(block.center.get_field(FI.VX_STAR))
    assert not math.isfinite(block.center.get_field(FI.P_NEXT))
    
    # Execute the sync operator
    sync_ghost_trial_buffers(block)
    
    # Post-condition check: Trial must exactly match Foundation
    assert block.center.get_field(FI.VX_STAR) == block.center.get_field(FI.VX)
    assert block.center.get_field(FI.VY_STAR) == block.center.get_field(FI.VY)
    assert block.center.get_field(FI.VZ_STAR) == block.center.get_field(FI.VZ)
    assert block.center.get_field(FI.P_NEXT) == block.center.get_field(FI.P)
    
    # Final check: No NaNs/Infs remain
    assert math.isfinite(block.center.get_field(FI.VX_STAR))
    assert math.isfinite(block.center.get_field(FI.P_NEXT))

def test_ghost_sync_preserves_foundation(ghost_stencil):
    """
    Rule 5 Verification: Ensures syncing trial buffers does not corrupt 
    the underlying Foundation (SSoT check).
    """
    block = ghost_stencil
    original_p = block.center.get_field(FI.P)
    
    sync_ghost_trial_buffers(block)
    
    # The foundation must remain untouched
    assert block.center.get_field(FI.P) == original_p