# tests/quality_gates/physics_gate/test_forces.py

import pytest

from src.step3.ops.forces import get_local_body_force
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy


def setup_force_block(block, force_tuple):
    """
    Manually wires the force vector into the StencilBlock's protected slot.
    This mimics the Step 2 Assembly logic where state.config values 
    are mapped to the object graph.
    """
    # Targets the internal slot to satisfy Rule 4 (Hierarchy over Convenience)
    # This ensures the function is actually reading the SSoT property.
    object.__setattr__(block, '_f_vals', force_tuple)
    return block

# --- Scenario 1: The Vacuum Gate (Zero Gravity) ---
def test_body_force_zero_gravity():
    """Verifies that a null force configuration returns a clean zero vector."""
    block = make_step3_output_dummy()
    expected_forces = (0.0, 0.0, 0.0)
    setup_force_block(block, expected_forces)
    
    result = get_local_body_force(block)
    
    assert result == expected_forces
    assert isinstance(result, tuple)

# --- Scenario 2: Earth Gravity (Standard Fy) ---
def test_body_force_earth_gravity():
    """Verifies standard downward vertical force mapping."""
    block = make_step3_output_dummy()
    # Fx=0, Fy=-9.81, Fz=0
    expected_forces = (0.0, -9.81, 0.0)
    setup_force_block(block, expected_forces)
    
    result = get_local_body_force(block)
    
    assert result == expected_forces
    # Rule 7 Check: Ensure precision is maintained for floating point constants
    assert result[1] == -9.81

# --- Scenario 3: Complex Multi-Axis Force ---
def test_body_force_diagonal_flow():
    """Verifies that non-standard, multi-axis forces are preserved without drift."""
    block = make_step3_output_dummy()
    # A complex force vector likely from a custom simulation config
    expected_forces = (0.5, 1.2, -0.3)
    setup_force_block(block, expected_forces)
    
    result = get_local_body_force(block)
    
    assert result == expected_forces

# --- Scenario 4: Precision Integrity Gate ---
def test_body_force_high_precision():
    """Ensures no truncation occurs between the config and the operator."""
    block = make_step3_output_dummy()
    # High precision value to test for any unintended casting to float32 early
    val = 1.23456789012345
    expected_forces = (val, val, val)
    setup_force_block(block, expected_forces)
    
    result = get_local_body_force(block)
    
    # We use pytest.approx to confirm machine precision (Rule 7)
    assert result[0] == pytest.approx(val, abs=1e-15)