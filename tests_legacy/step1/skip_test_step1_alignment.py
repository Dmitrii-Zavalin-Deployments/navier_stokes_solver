# tests/step1/test_step1_alignment.py

import numpy as np
import pytest

from src.solver_input import SolverInput
from src.step1.orchestrate_step1 import orchestrate_step1
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy

"""
test_step1_alignment.py
Constitutional Role: The Step 1 Alignment Auditor.
Compliance: Phase C, Section 3 (Unit Test: Orchestration Alignment).
"""

def test_step1_alignment_logic_to_frozen_truth():
    """
    VERIFICATION TASK: orchestrate_step1(input_obj) == step1_output_dummy.
    """
    # 1. Input: Hydrate the Object from the Dummy Dict
    raw_json = solver_input_schema_dummy()
    input_obj = SolverInput.from_dict(raw_json)
    
    # 2. Execution: Run the actual orchestration logic
    result_state = orchestrate_step1(input_obj, iteration=0, time=0.0)
    
    # 3. Reference: The 'Frozen Truth'
    expected_state = make_step1_output_dummy(nx=2, ny=2, nz=2)
    
    # FIX: Use dot notation for object access
    expected_state.masks.mask = np.asarray(input_obj.mask.data).reshape((2,2,2), order="F")

    # --- AUDIT A: GRID & SPATIAL PARAMETERS ---
    # Access result_state properties directly (no dict strings)
    assert result_state.grid.nx == expected_state.grid.nx
    assert np.isclose(result_state.grid.dx, expected_state.grid.dx)
    assert np.isclose(result_state.grid.dy, expected_state.grid.dy)
    assert np.isclose(result_state.grid.dz, expected_state.grid.dz)

    # --- AUDIT B: STAGGERED MEMORY LAYOUT ---
    assert result_state.fields.U.shape == expected_state.fields.U.shape
    assert result_state.fields.V.shape == expected_state.fields.V.shape
    assert result_state.fields.W.shape == expected_state.fields.W.shape
    assert result_state.fields.P.shape == expected_state.fields.P.shape

    # --- AUDIT C: TOPOLOGY & MASKING ---
    # Result mask is a 3D numpy array in the new architecture
    assert isinstance(result_state.masks.mask, np.ndarray)
    assert result_state.masks.mask.shape == (2, 2, 2)
    assert np.array_equal(result_state.masks.mask, expected_state.masks.mask)

    # --- AUDIT D: BOUNDARY CONDITIONS ---
    assert len(result_state.boundary_lookup) == 6
    for face in result_state.boundary_lookup.values():
        assert "type" in face
        assert "u" in face
        assert "p" in face

    # --- AUDIT E: CONSTANT PROPAGATION ---
    # FIX: Access via dot notation
    assert result_state.fluid.rho == input_obj.fluid_properties.density
    assert result_state.fluid.mu == input_obj.fluid_properties.viscosity

def test_step1_sensitivity_firewall():
    """
    Verifies that invalid data triggers a ValueError from the setter.
    """
    invalid_input = solver_input_schema_dummy()
    invalid_input["grid"]["nx"] = -1 
    
    # FIX: Expect ValueError (raised by GridInput.nx setter)
    with pytest.raises(ValueError) as excinfo:
        SolverInput.from_dict(invalid_input)
    
    assert "nx must be >= 1" in str(excinfo.value)