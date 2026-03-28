# tests/quality_gates/contract_gate/test_contract_integrity.py

import numpy as np
import pytest

from src.common.simulation_context import SimulationContext
from src.step1.orchestrate_step1 import orchestrate_step1
from tests.helpers.solver_input_schema_dummy import create_validated_input


def wrap_in_context(solver_input):
    """
    Compliance: Rule 4. Wraps raw data in the appropriate 
    SimulationContext to avoid __slots__ AttributeErrors.
    """
    return SimulationContext(input_data=solver_input, config=None)

def test_gate_4_type_casting_integrity():
    """
    Verification: Ensure orchestrate_step1 casts string-based JSON intake.
    Asserts: Type conversion and value accuracy (Atomic Truth).
    """
    solver_input = create_validated_input(nx=2)
    context = wrap_in_context(solver_input)
    
    # Bypass setters to inject 'stringified' JSON data for testing
    context.input_data.grid._nx = "2"
    context.input_data.fluid_properties._density = "1.225"
    
    # Execute Step 1 mapping
    state = orchestrate_step1(context)
    
    # Assertions: Verify the 'No-Garbage' Principle via casting
    assert isinstance(state.grid.nx, int), f"Casting Failure: nx is {type(state.grid.nx)}"
    assert state.grid.nx == 2
    assert isinstance(state.fluid_properties.density, float)
    assert np.isclose(state.fluid_properties.density, 1.225)

def test_gate_4_schema_validation_and_firewall():
    """
    Verification: Ensure state.validate_against_schema("schema/solver_input_schema.json") blocks incomplete states.
    Asserts: Validation fails on uninitialized/None fields (Double-Lock Barrier).
    """
    solver_input = create_validated_input()
    context = wrap_in_context(solver_input)
    state = orchestrate_step1(context)
    
    # 1. Corrupt the state foundation
    state.grid._nx = None 
    
    # 2. Assert: The schema validator must catch the drift
    # This aligns with the 'Double-Lock' requirement in Line 74
    with pytest.raises(ValueError, match='Validation Failure'): # Replace with specific jsonschema.ValidationError if imported
        state.validate_against_schema("schema/solver_input_schema.json")
        
    # 3. Assert: ready_for_time_loop remains False (Sentinel Integrity)
    assert state.ready_for_time_loop is False

def test_gate_4_physical_drift_rollback():
    """
    Verification: Ensure property setters in solver_state.py protect existing data.
    Asserts: State preservation after failed "garbage" injection.
    """
    solver_input = create_validated_input()
    context = wrap_in_context(solver_input)
    state = orchestrate_step1(context)
    
    original_density = state.fluid_properties.density
    
    # Attempt to inject garbage
    with pytest.raises(TypeError):
        state.fluid_properties.density = "NOT_A_NUMBER"
        
    # Assert: The "No-Garbage" Principle preserved the atomic truth
    assert state.fluid_properties.density == original_density

def test_gate_4_numerical_explosion_audit():
    """
    Verification: Ensure audit_physical_bounds identifies NaNs in the FieldManager.
    Asserts: Detection of corrupted memory buffers before high-compute steps.
    """
    solver_input = create_validated_input()
    context = wrap_in_context(solver_input)
    state = orchestrate_step1(context)
    
    # Inject a Numerical Storm into the raw buffer
    state.fields.data[0, 0] = np.nan
    
    # Assert: High-compute kernels (Step 3) are protected by a crash
    with pytest.raises(ArithmeticError, match="NaN/Inf detected"):
        state.audit_physical_bounds()
    
    # Verify the state is correctly identified as dirty
    assert np.isnan(state.fields.data).any()