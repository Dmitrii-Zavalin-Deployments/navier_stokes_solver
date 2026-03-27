# tests/quality_gates/contract_gate/test_contract_integrity.py

import numpy as np
import pytest

from src.common.simulation_context import SimulationContext
from src.step1.orchestrate_step1 import orchestrate_step1
from tests.helpers.solver_input_schema_dummy import create_validated_input


def wrap_in_context(solver_input):
    """
    Helper to satisfy Rule 0 and Rule 4.
    SimulationContext is the valid container for 'input_data'.
    """
    return SimulationContext(input_data=solver_input, config=None)

def test_gate_4_type_casting_integrity():
    """
    Verification: Ensure orchestrate_step1 correctly casts string-based numbers 
    from JSON context into the specific types (int/float) required by the 
    Memory-Hardened Managers.
    Compliance: Rule 5 (Deterministic Initialization) & Rule 7 (Atomic Truth).
    """
    
    # 1. Setup: Create input and wrap it in the proper context container
    solver_input = create_validated_input(nx=2)
    context = wrap_in_context(solver_input)
    
    # Simulate raw JSON intake where types are often stringified
    # Note: We access via context.input_data to follow the SSoT hierarchy
    context.input_data.grid.nx = "10"
    context.input_data.fluid_properties.density = "1000.0"
    
    # 2. Execute Step 1: Orchestration/Mapping
    state = orchestrate_step1(context)
    
    # 3. Audit structural compliance (SSoT Rule 4)
    assert isinstance(state.grid.nx, int), "Gate Breach: nx failed to cast to int"
    assert state.grid.nx == 10
    
    assert isinstance(state.fluid_properties.density, float), "Gate Breach: density failed to cast to float"
    assert np.isclose(state.fluid_properties.density, 1000.0)

def test_gate_4_drift_detection_and_safety():
    """
    Gate 4: Contract Drift & "No-Garbage" Principle.
    Verification: Ensure that if a manager is corrupted or uninitialized, 
    the property setters/getters in src/common/solver_state.py trigger the firewall.
    Compliance: Rule 0 (__slots__ protection) & Rule 7 (Numerical Truth).
    """
        
    solver_input = create_validated_input()
    context = wrap_in_context(solver_input)
    state = orchestrate_step1(context)
    
    # 1. Test "Department Safe" Protection (ValidatedContainer logic)
    with pytest.raises(TypeError):
        state.fluid_properties.density = "NOT_A_FLOAT"

    # 2. Test Physical Bound Enforcement (Rule 7 logic)
    with pytest.raises(ValueError, match="Density must be > 0"):
        state.fluid_properties.density = -5.0

def test_gate_4_foundation_readiness_lock():
    """
    Verification: Ensure ready_for_time_loop sentinel executes 
    verify_foundation_integrity and validate_physical_readiness.
    Compliance: Rule 2 (Zero-Debt / Pre-flight logic).
    """
        
    solver_input = create_validated_input()
    context = wrap_in_context(solver_input)
    state = orchestrate_step1(context)
    
    # 1. Break the 'readiness' contract (Rule 5: No Structural Fallbacks)
    state.grid.nx = None
    
    # 2. Action: Setting ready_for_time_loop to True triggers the pre-flight audit.
    with pytest.raises(RuntimeError, match="Grid not properly initialized"):
        state.ready_for_time_loop = True

def test_gate_4_vectorized_audit_grounding():
    """
    Verification: Ensure audit_physical_bounds correctly identifies 
    Numerical Explosions (NaNs) in the Foundation buffer.
    Compliance: Rule 1 (Field Precision Audit) & Rule 7 (Numerical Truth).
    """
        
    solver_input = create_validated_input()
    context = wrap_in_context(solver_input)
    state = orchestrate_step1(context)
    
    # 1. Inject a 'Numerical Storm' (NaN) into the monolithic Foundation (Rule 1)
    state.fields.data[0, 0] = np.nan
    
    # 2. Verification: The audit must detect this breach.
    with pytest.raises(ArithmeticError, match="NaN/Inf detected"):
        state.audit_physical_bounds()