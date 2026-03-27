# tests/quality_gates/contract_gate/test_contract_integrity.py

import pytest
import numpy as np
from src.step1.orchestrate_step1 import orchestrate_step1
from src.common.solver_state import SolverState

def test_gate_4_type_casting_integrity(solver_input_schema_dummy):
    """
    Verification: Ensure orchestrate_step1 correctly casts string-based numbers 
    from JSON context into the specific types (int/float) required by the 
    Memory-Hardened Managers.
    """
    # 1. Setup context with stringified numbers (simulating raw JSON intake)
    context = solver_input_schema_dummy
    context.input_data.grid.nx = "10"
    context.input_data.fluid_properties.density = "1000.0"
    
    # 2. Execute Step 1: Mapping
    state = orchestrate_step1(context)
    
    # 3. Audit structural compliance
    assert isinstance(state.grid.nx, int), "Gate Breach: nx failed to cast to int"
    assert state.grid.nx == 10
    assert isinstance(state.fluid_properties.density, float), "Gate Breach: density failed to cast to float"
    assert np.isclose(state.fluid_properties.density, 1000.0)

def test_gate_4_drift_detection_and_safety(solver_input_schema_dummy):
    """
    Gate 4: Contract Drift & "No-Garbage" Principle.
    Verification: Ensure that if a manager is corrupted or uninitialized, 
    the property setters/getters in src/common/solver_state.py trigger the firewall.
    """
    context = solver_input_schema_dummy
    state = orchestrate_step1(context)
    
    # 1. Test "Department Safe" Protection (ValidatedContainer logic)
    # Attempting to set an invalid type should trigger the setter firewall
    with pytest.raises(TypeError):
        state.fluid_properties.density = "NOT_A_FLOAT"

    # 2. Test Physical Bound Enforcement (Rule 7 logic)
    # The setter for FluidPropertiesManager enforces density > 0
    with pytest.raises(ValueError, match="Density must be > 0"):
        state.fluid_properties.density = -5.0

def test_gate_4_foundation_readiness_lock(solver_input_schema_dummy):
    """
    Verification: Ensure ready_for_time_loop sentinel executes 
    verify_foundation_integrity and validate_physical_readiness.
    """
    context = solver_input_schema_dummy
    state = orchestrate_step1(context)
    
    # Simulate an uninitialized grid to break the 'readiness' contract
    state.grid.nx = None
    
    # Setting ready_for_time_loop to True triggers the full pre-flight audit
    with pytest.raises(RuntimeError, match="Grid not properly initialized"):
        state.ready_for_time_loop = True

def test_gate_4_vectorized_audit_grounding(solver_input_schema_dummy):
    """
    Verification: Ensure audit_physical_bounds correctly identifies 
    Numerical Explosions (NaNs) in the Foundation buffer.
    """
    state = orchestrate_step1(solver_input_schema_dummy)
    
    # Inject a 'Numerical Storm' (NaN) into the monolithic buffer
    state.fields.data[0, 0] = np.nan
    
    # The audit must detect this breach of the "No-Garbage" Principle
    with pytest.raises(ArithmeticError, match="NaN/Inf detected"):
        state.audit_physical_bounds()