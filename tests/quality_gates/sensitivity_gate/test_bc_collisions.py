import logging
import pytest
from src.step3.boundaries.dispatcher import get_applicable_boundary_configs
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy

# Tracking the specific dispatcher logger
DISPATCH_LOGGER = "src.step3.boundaries.dispatcher"

def test_gate_3a_3b_dispatcher_mask_symmetry(caplog):
    """
    Gate 3.A & 3.B: Dispatcher-Mask Symmetry Audit.
    Verification: Ensure mask values (-1, 0) trigger the correct Axiomatic logic.
    Compliance: Physical Logic Firewall (Rule 4).
    """
    # 1. Setup: Set caplog to capture DEBUG for the dispatcher specifically
    caplog.set_level(logging.DEBUG, logger=DISPATCH_LOGGER)
    state = make_step2_output_dummy(nx=4, ny=4, nz=4)
    
    boundary_cfg = [
        {"location": "wall", "type": "no-slip", "values": {"u": 0.1}} # Distinctive value
    ]
    domain_cfg = {"type": "INTERNAL"}
    block = state.stencil_matrix[10] # Guaranteed interior core block

    # --- Audit Step 3.A: User-Defined Wall Logic (-1) ---
    block.center.mask = -1
    rules = get_applicable_boundary_configs(block, boundary_cfg, state.grid, domain_cfg)
    
    assert "DISPATCH [Mask]" in caplog.text
    assert "treated as Wall (mask -1)" in caplog.text
    assert rules[0]["values"]["u"] == 0.1, "Wall should follow User-Defined boundary_cfg"
    
    caplog.clear()

    # --- Audit Step 3.B: Axiomatic Solid Logic (0) ---
    block.center.mask = 0
    rules = get_applicable_boundary_configs(block, boundary_cfg, state.grid, domain_cfg)
    
    assert "DISPATCH [Mask]" in caplog.text
    assert "treated as Solid (mask 0)" in caplog.text
    # Verification of the hardcoded Axiom: Should be 0.0 regardless of boundary_cfg
    assert rules[0]["location"] == "solid"
    assert rules[0]["values"]["u"] == 0.0, "Solid Axiom Failure: Must be hardcoded 0.0"

def test_gate_3c_interior_fluid_axiom():
    """
    Gate 3.C: Interior Fluid Axiom Audit.
    Verification: Ensure mask 1 returns 'interior' type with no enforced values.
    """
    state = make_step2_output_dummy(nx=4, ny=4, nz=4)
    block = state.stencil_matrix[10]
    block.center.mask = 1 # Fluid
    
    rules = get_applicable_boundary_configs(block, [], state.grid, {"type": "INTERNAL"})
    
    assert rules[0]["location"] == "interior"
    assert rules[0]["type"] == "fluid_gas"
    assert rules[0]["values"] == {}, "Interior Axiom Failure: Should not enforce values"

def test_gate_3d_external_flow_axiom():
    """
    Gate 3.D: External Flow (Far-field) Axiom Audit.
    Verification: Ensure domain 'EXTERNAL' forces free-stream velocity on ghosts.
    """
    state = make_step2_output_dummy(nx=4, ny=4, nz=4)
    # Find a block that borders a ghost (e.g., index 0 usually borders x_min ghost)
    block = state.stencil_matrix[0] 
    
    domain_cfg = {
        "type": "EXTERNAL",
        "reference_velocity": [10.0, 0.0, 0.0]
    }
    
    rules = get_applicable_boundary_configs(block, [], state.grid, domain_cfg)
    
    assert rules[0]["type"] == "free-stream"
    assert rules[0]["values"]["u"] == 10.0, "External Axiom Failure: Reference velocity ignored"

def test_gate_3a_missing_wall_config_collision():
    """
    Step 3.A: Catch KeyError when mask -1 is present but 'wall' config is missing.
    Ensures that for walls (unlike solids), the user MUST provide config.
    """
    state = make_step2_output_dummy(nx=4, ny=4, nz=4)
    block = state.stencil_matrix[10]
    block.center.mask = -1
    
    # Empty boundary config list
    incomplete_cfg = []
    
    with pytest.raises(KeyError, match="No boundary configuration found for location: 'wall'"):
        get_applicable_boundary_configs(block, incomplete_cfg, state.grid, {"type": "INTERNAL"})