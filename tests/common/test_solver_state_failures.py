# tests/common/test_solver_state_failures.py

import pytest
import numpy as np
from unittest.mock import MagicMock
from src.common.solver_state import (
    SolverState, DomainManager, GridManager, FluidPropertiesManager, 
    BoundaryCondition, MaskManager, ExternalForceManager, verify_foundation_integrity
)

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
    # Line 513: Uninitialized Mask Serialization
    mm = MaskManager()
    with pytest.raises(RuntimeError, match="uninitialized"):
        mm.to_dict()
    
    # Line 558: Uninitialized Force Serialization
    efm = ExternalForceManager()
    with pytest.raises(AttributeError, match="force_vector must be initialized"):
        efm.to_dict()
        
    # Line 576: Invalid force vector size
    with pytest.raises(ValueError, match="3D NumPy array"):
        efm.force_vector = np.array([9.8])

# --- 4. Testing Audit/Readiness Failures (Lines 606, 610, 613) ---
def test_validate_physical_readiness_failures():
    state = SolverState()
    # Line 606: Missing constraints
    state.fields = MagicMock()
    state.fields.data = np.zeros((10, 10))
    state.physical_constraints = None
    with pytest.raises(RuntimeError, match="Physical Constraints are not defined"):
        state.validate_physical_readiness()
        
    # Setup constraints for next checks
    from src.common.solver_state import PhysicalConstraintsManager
    state.physical_constraints = PhysicalConstraintsManager()
    
    # Line 610: NaNs in Foundation
    state.fields.data[0, 0] = np.nan
    with pytest.raises(RuntimeError, match="NaNs/Infs detected"):
        state.validate_physical_readiness()
        
    # Line 613: Grid not initialized
    state.fields.data[0, 0] = 0.0
    state.grid = GridManager() # nx is None
    with pytest.raises(RuntimeError, match="Grid not properly initialized"):
        state.validate_physical_readiness()