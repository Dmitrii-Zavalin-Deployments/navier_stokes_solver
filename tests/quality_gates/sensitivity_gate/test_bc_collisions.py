# tests/quality_gates/sensitivity_gate/test_bc_collisions.py

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
    """
    # Set caplog to capture the specific dispatcher logger at DEBUG level
    caplog.set_level(logging.DEBUG, logger="src.step3.boundaries.dispatcher")
    
    # 1. Setup: Use Step 2 Dummy for a hydrated 4x4x4 core
    state = make_step2_output_dummy(nx=4, ny=4, nz=4)
    
    # FIX: Provide full face definitions to prevent 'x_min' KeyError 
    # while auditing internal mask logic.
    boundary_cfg = [
        {"location": "wall", "type": "no-slip", "values": {}},
        {"location": "solid", "type": "no-slip", "values": {}},
        {"location": "x_min", "type": "no-slip", "values": {}},
        {"location": "x_max", "type": "no-slip", "values": {}},
        {"location": "y_min", "type": "no-slip", "values": {}},
        {"location": "y_max", "type": "no-slip", "values": {}},
        {"location": "z_min", "type": "no-slip", "values": {}},
        {"location": "z_max", "type": "no-slip", "values": {}}
    ]
    domain_cfg = {"type": "INTERNAL"}

    # We pick an interior core block. In a 4x4x4 core (indices 1-4), 
    # index [10] is roughly (3,2,1), far enough from some faces but 
    # the dummy config above covers all spatial contingencies.
    block = state.stencil_matrix[10] 

    # 2. Audit Step 3.A: Wall Logic (-1)
    block.center.mask = -1
    get_applicable_boundary_configs(block, boundary_cfg, state.grid, domain_cfg)
    
    assert "DISPATCH [Mask]" in caplog.text, "Symmetry Breach: Missing DISPATCH [Mask] log for wall."
    assert "treated as Wall (mask -1)" in caplog.text
    
    caplog.clear()

    # 3. Audit Step 3.B: Solid Logic (0)
    block.center.mask = 0
    get_applicable_boundary_configs(block, boundary_cfg, state.grid, domain_cfg)
    
    assert "DISPATCH [Mask]" in caplog.text, "Symmetry Breach: Missing DISPATCH [Mask] log for solid."
    assert "treated as Solid (mask 0)" in caplog.text

def test_gate_3a_missing_wall_config_collision():
    """
    Step 3.A: Catch KeyError when mask -1 is present but 'wall' config is missing.
    """
    state = make_step2_output_dummy(nx=4, ny=4, nz=4)
    block = state.stencil_matrix[10]
    block.center.mask = -1
    
    # We provide the face configs but EXCLUDE 'wall' to trigger the specific KeyError.
    incomplete_cfg = [
        {"location": "x_min", "type": "no-slip", "values": {}},
        {"location": "x_max", "type": "no-slip", "values": {}},
        {"location": "y_min", "type": "no-slip", "values": {}},
        {"location": "y_max", "type": "no-slip", "values": {}},
        {"location": "z_min", "type": "no-slip", "values": {}},
        {"location": "z_max", "type": "no-slip", "values": {}}
    ]
    
    # Verification: Ensure the failure is about 'wall', not a spatial face.
    with pytest.raises(KeyError, match="No boundary configuration found for location: 'wall'"):
        get_applicable_boundary_configs(block, incomplete_cfg, state.grid, {"type": "INTERNAL"})