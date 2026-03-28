import logging

import pytest

from src.step3.boundaries.dispatcher import get_applicable_boundary_configs
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy

logger = logging.getLogger(__name__)

def test_gate_3a_3b_dispatcher_mask_symmetry(caplog):
    """
    Gate 3.A & 3.B: Dispatcher-Mask Symmetry Audit
    Verification: Ensure mask values (-1, 0) trigger explicit log dispatching.
    Compliance: Physical Logic Firewall (Rule 4).
    
    Mandate: The solver must log 'DISPATCH [Mask]' for material transitions.
    """
    # Set caplog to capture the specific dispatcher logger at DEBUG level
    caplog.set_level(logging.DEBUG, logger="src.step3.boundaries.dispatcher")
    
    # 1. Setup: Use Step 2 Dummy for a hydrated 4x4x4 core (6x6x6 memory)
    # This ensures we have a real StencilBlock with neighbor topology.
    state = make_step2_output_dummy(nx=4, ny=4, nz=4)
    boundary_cfg = [{"location": "wall", "type": "no-slip", "values": {}}]
    domain_cfg = {"type": "INTERNAL"}

    # Select an interior core block to isolate Mask logic from Spatial (Ghost) logic
    block = state.stencil_matrix[0] 

    # 2. Audit Step 3.A: Wall Logic (-1)
    block.center.mask = -1
    get_applicable_boundary_configs(block, boundary_cfg, state.grid, domain_cfg)
    
    assert "DISPATCH [Mask]" in caplog.text, "Symmetry Breach: Missing DISPATCH [Mask] log for wall."
    assert "treated as Wall (mask -1)" in caplog.text, "Logic Breach: Mask -1 did not identify as Wall."
    
    caplog.clear()

    # 3. Audit Step 3.B: Solid Logic (0)
    block.center.mask = 0
    get_applicable_boundary_configs(block, boundary_cfg, state.grid, domain_cfg)
    
    assert "DISPATCH [Mask]" in caplog.text, "Symmetry Breach: Missing DISPATCH [Mask] log for solid."
    assert "treated as Solid (mask 0)" in caplog.text, "Logic Breach: Mask 0 did not identify as Solid."

def test_gate_3a_missing_wall_config_collision():
    """
    Step 3.A Verification Strategy: Catch KeyError when mask -1 is present 
    but no 'wall' configuration exists in the user-provided JSON.
    """
    state = make_step2_output_dummy(nx=4, ny=4, nz=4)
    block = state.stencil_matrix[0]
    block.center.mask = -1
    
    # Compliance: Rule 5 (Strict Registry). Fail-fast if the key is missing.
    with pytest.raises(KeyError, match="No boundary configuration found for location: 'wall'"):
        get_applicable_boundary_configs(block, [], state.grid, {"type": "INTERNAL"})