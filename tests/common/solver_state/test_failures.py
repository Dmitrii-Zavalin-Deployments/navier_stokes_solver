# tests/common/solver_state/test_failures.py

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.common.solver_state import (
    BoundaryCondition,
    DomainManager,
    ExternalForceManager,
    FieldManager,
    FluidPropertiesManager,
    GridManager,
    MaskManager,
    PhysicalConstraintsManager,
    SolverState,
    verify_foundation_integrity,
)
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy


# --- 1. Testing Foundation Integrity Failures (Lines 24-25, 56) ---
def test_verify_foundation_integrity_failures():
    state = MagicMock()
    # Line 24-25: Fields or data is None
    state.fields = None
    with pytest.raises(RuntimeError, match="Fields buffer not initialized"):
        verify_foundation_integrity(state)
        
    # Line 56: Stencil count > Buffer size (Overflow)
    state.fields = MagicMock()
    state.fields.data = np.zeros((10, 5))
    state.stencil_matrix = [MagicMock()] * 11 # 11 > 10
    with pytest.raises(RuntimeError, match="Architecture integrity compromised"):
        verify_foundation_integrity(state)

# --- 2. Testing Manager Validation Failures (Lines 127, 135, 237, 328, 340, 346, 358) ---
def test_manager_validation_errors():
    dm = DomainManager()
    # Line 127: Invalid Domain Type
    with pytest.raises(ValueError, match="Invalid domain type"):
        dm.type = "GHOST_ZONE"
    
    # Line 135: Invalid Reference Velocity (Wrong Type/Size)
    with pytest.raises(TypeError):
        dm.reference_velocity = "fast"
    with pytest.raises(TypeError):
        dm.reference_velocity = np.array([1, 0]) # 2D instead of 3D

    # Line 237: Negative Fluid Properties
    fp = FluidPropertiesManager()
    with pytest.raises(ValueError, match="Density must be > 0"):
        fp.density = -1.0
    with pytest.raises(ValueError, match="Viscosity must be >= 0"):
        fp.viscosity = -0.5

    # Line 328: Invalid Boundary Location
    bc = BoundaryCondition()
    with pytest.raises(ValueError, match="Invalid location"):
        bc.location = "center_of_earth"
        
    # Line 340: Invalid Boundary Type
    with pytest.raises(ValueError, match="Invalid type"):
        bc.type = "teleportation"

    # Line 346: Boundary Values not a dict
    with pytest.raises(TypeError, match="values must be a dict"):
        bc.values = [0, 0, 0]

    # Line 358: Boundary conditions not a list
    from src.common.solver_state import BoundaryConditionManager
    bcm = BoundaryConditionManager()
    with pytest.raises(TypeError, match="Must be a list"):
        bcm.conditions = "not_a_list"

# --- 3. Testing Mask and Force Failures (Lines 513, 558, 576) ---
def test_mask_and_force_failures():
    # 1. Line 513: Uninitialized Mask Serialization
    mm = MaskManager()
    # Matches the direct RuntimeError in MaskManager.to_dict (Line 328)
    with pytest.raises(RuntimeError, match="MaskManager: _mask is uninitialized"):
        mm.to_dict()
    
    # 2. Line 558: Uninitialized Force Serialization
    efm = ExternalForceManager()
    # Matches the _get_safe error from BaseContainer
    with pytest.raises(RuntimeError, match="Access Error: 'force_vector' in ExternalForceManager is uninitialized"):
        efm.to_dict()
        
    # 3. Line 576: Invalid force vector size
    with pytest.raises(ValueError, match="force_vector must be a 3D NumPy array"):
        efm.force_vector = np.array([9.8])

def test_validate_physical_readiness_failures():
    state = SolverState()
    
    fm = FieldManager()
    fm.data = np.zeros((10, 10))
    state.fields = fm 
    
    # Line 609: Missing constraints triggers _get_safe before Line 610
    state.physical_constraints = None
    with pytest.raises(RuntimeError, match="Access Error: 'physical_constraints' in SolverState is uninitialized"):
        state.validate_physical_readiness()
        
    # Setup constraints for next checks
    state.physical_constraints = PhysicalConstraintsManager()
    
    # Line 612: NaNs in Foundation
    fm.data[0, 0] = np.nan
    with pytest.raises(RuntimeError, match="CRITICAL: NaNs/Infs detected in Foundation buffer!"):
        state.validate_physical_readiness()
        
    # Line 614 in src/common/solver_state.py
    fm.data[0, 0] = 0.0 
    state.grid = GridManager() # _nx is None by default
    
    # We expect the BaseContainer's Access Error because accessing 'self.grid.nx' 
    # triggers _get_safe("nx") before the 'is None' check can complete.
    with pytest.raises(RuntimeError, match="Access Error: 'nx' in GridManager is uninitialized"):
        state.validate_physical_readiness()

def test_initial_condition_velocity_validation():
    """
    Validates Lines 236-238 using the Step 1 Dummy for context.
    Ensures that velocity must be a 3D NumPy array.
    """
    # Use the dummy to get a pre-hydrated manager
    state = make_step1_output_dummy(nx=2, ny=2, nz=2)
    ic = state.initial_conditions 

    # 1. Test non-numpy array input
    with pytest.raises(ValueError, match="Velocity must be a 3D NumPy array."):
        ic.velocity = [0.0, 0.0, 0.0] 

    # 2. Test wrong size (2D)
    with pytest.raises(ValueError, match="Velocity must be a 3D NumPy array."):
        ic.velocity = np.array([1.0, 0.0])

    # 3. Success Case verification
    valid_v = np.array([0.5, 0.5, 0.5])
    ic.velocity = valid_v
    np.testing.assert_array_equal(ic.velocity, valid_v)

def test_mask_validation_logic():
    """
    Validates Lines 339-341: Enforces Mask must be a NumPy array containing only -1, 0, or 1.
    """
    from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy
    
    # 1. Hydrate via Dummy
    state = make_step1_output_dummy(nx=2, ny=2, nz=2)
    mm = state.mask

    # 2. Test non-array input (Trigger first part of the 'or')
    with pytest.raises(ValueError, match="Mask must be a NumPy array of -1, 0, 1."):
        mm.mask = [1, 0, -1]  # List instead of np.ndarray

    # 3. Test invalid values (Trigger second part of the 'or')
    with pytest.raises(ValueError, match="Mask must be a NumPy array of -1, 0, 1."):
        mm.mask = np.array([1, 0, 2])  # '2' is not a valid cell type

    # 4. Test floats (Strictly enforcing discrete topology)
    with pytest.raises(ValueError, match="Mask must be a NumPy array of -1, 0, 1."):
        mm.mask = np.array([1.0, 0.5, -1.0])

    # 5. Success Case: Valid discrete topology
    valid_mask = np.array([-1, 0, 1], dtype=int)
    mm.mask = valid_mask
    np.testing.assert_array_equal(mm.mask, valid_mask)
    
    # 6. Test None (Should pass bypass check)
    mm.mask = None
    assert mm._mask is None

def test_external_force_serialization_enforcement():
    """
    Validates Lines 345-347: Ensures force_vector is initialized before serialization.
    Note: BaseContainer's _get_safe (Rule 5) catches this first as a RuntimeError.
    """

    efm = ExternalForceManager()

    # 1. Test Access Error on uninitialized serialization
    # The base class _get_safe raises RuntimeError if any slot is None.
    with pytest.raises(RuntimeError, match="Access Error: 'force_vector' in ExternalForceManager is uninitialized."):
        efm.to_dict()

    # 2. Test validation logic in the setter
    with pytest.raises(ValueError, match="force_vector must be a 3D NumPy array."):
        efm.force_vector = np.array([0.0, -9.81]) 

    # 3. Success Case: Valid serialization
    gravity = np.array([0.0, 0.0, -9.81])
    efm.force_vector = gravity
    
    serialized = efm.to_dict()
    assert serialized["force_vector"] == [0.0, 0.0, -9.81]

def test_rollback_without_cache_fails():
    """
    Validates Lines 512-514: Ensures rollback_to_stable_state raises RuntimeError 
    if capture_stable_state was never called.
    """

    # 1. Initialize a clean state
    state = SolverState()
    
    # 2. Verify cache is explicitly None on init
    assert state._cache_buffer is None

    # 3. Trigger the safety sentinel (The "Smoking Gun")
    # This simulates a failure occurring before the first iteration's capture
    with pytest.raises(RuntimeError, match="CRITICAL: Rollback requested but no cache exists."):
        state.rollback_to_stable_state()

    # 4. Verify that after a capture, rollback no longer raises
    # (Mocking minimal requirements for capture)
    from src.common.solver_state import FieldManager
    state.fields = FieldManager()
    state.fields.allocate(n_cells=10)
    
    state.capture_stable_state()
    assert state._cache_buffer is not None
    
    # This should now pass without error
    state.rollback_to_stable_state()