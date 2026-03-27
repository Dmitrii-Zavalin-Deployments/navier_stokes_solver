# tests/quality_gates/sensitivity_gate/test_bc_collisions.py

import pytest
import logging
from src.step3.boundaries.dispatcher import get_applicable_boundary_configs

def test_gate_3a_3b_dispatcher_mask_symmetry(caplog):
    """
    Gate 3.A & 3.B: Dispatcher-Mask Symmetry Audit
    Verification: Ensure mask values (-1, 0) trigger explicit log dispatching.
    Compliance: Physical Logic Firewall Mandate.
    """
    # Set caplog to capture DEBUG level from the specific dispatcher logger
    caplog.set_level(logging.DEBUG, logger="src.step3.boundaries.dispatcher")
    
    # 1. Setup Mock Objects
    class MockCell: 
        def __init__(self, mask): 
            self.mask = mask
            # Set indices to 1 to bypass the _get_domain_location_type (Step 1 spatial check)
            self.i, self.j, self.k = 1, 1, 1

    class MockBlock: 
        def __init__(self, mask): 
            self.center = MockCell(mask)
            self.id = "block_test_0"

    class MockGrid: 
        nx, ny, nz = 10, 10, 10

    # Minimum config required for mask -1 (Wall)
    boundary_cfg = [{"location": "wall", "type": "no-slip", "values": {}}]
    grid = MockGrid()
    domain_cfg = {"type": "INTERNAL"}

    # 2. Audit Step 3.A: Wall Logic (-1)
    get_applicable_boundary_configs(MockBlock(-1), boundary_cfg, grid, domain_cfg)
    assert "DISPATCH [Mask]" in caplog.text
    assert "treated as Wall (mask -1)" in caplog.text
    
    caplog.clear()

    # 3. Audit Step 3.B: Solid Logic (0)
    get_applicable_boundary_configs(MockBlock(0), boundary_cfg, grid, domain_cfg)
    assert "DISPATCH [Mask]" in caplog.text
    assert "treated as Solid (mask 0)" in caplog.text

def test_gate_3a_missing_wall_config_collision():
    """
    Verification Strategy: Catch KeyError when mask -1 is present but 
    no 'wall' configuration exists in JSON.
    """
    class MockCell: 
        def __init__(self, mask): self.mask, self.i, self.j, self.k = mask, 1, 1, 1
    class MockBlock: 
        def __init__(self, mask): self.center = MockCell(mask); self.id = "block_err"
    
    # Provide an empty boundary list to trigger the KeyError in _find_config
    with pytest.raises(KeyError, match="No boundary configuration found for location: 'wall'"):
        get_applicable_boundary_configs(MockBlock(-1), [], MockGrid(), {"type": "INTERNAL"})

class MockGrid:
    nx, ny, nz = 10, 10, 10