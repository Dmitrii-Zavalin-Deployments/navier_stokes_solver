# tests/quality_gates/sensitivity_gate/test_stencil_fidelity.py

import pytest
import numpy as np

from src.common.field_schema import FI
from src.step2.stencil_assembler import assemble_stencil_matrix
from tests.helpers.solver_state_dummy import create_dummy_state

def test_gate_2a_registry_traceability():
    """
    Gate 2.A: Stencil Registry Traceability Audit
    
    Verification: Physical constants (rho, mu) must propagate from SolverState 
    to StencilBlock objects during Step 2 assembly.
    Compliance: Rule 4 (SSoT) & Rule 5 (Deterministic Initialization).
    """

    # 1. Setup: Initialize a dummy state with explicit core dimensions
    state = create_dummy_state(nx=2, ny=2, nz=2)

    # Setup unique 'traceable' physical constants (Rule 7: Atomic Truth)
    # Using non-standard floats to ensure we aren't hitting default 1.0 or 0.0
    traceable_rho = 13.52
    traceable_mu = 0.00173
    
    # Correct SSoT access via sub-containers (Rule 4)
    state.fluid_properties.density = traceable_rho
    state.fluid_properties.viscosity = traceable_mu
    
    # 2. Action: Execute Step 2 Assembly
    # This triggers the Stencil Registry and physics_params extraction.
    # We pass the 'state' which contains both 'grid' and 'fluid_properties' (Rule 4).
    stencils = assemble_stencil_matrix(state)
    
    # 3. Verification: Ensure the Registry successfully wired the physics
    assert len(stencils) > 0, "Gate Breach: No stencils assembled."
    
    # Audit the first Core StencilBlock (logical index 0)
    sample_block = stencils[0]
    
    # Verification: Physical constant propagation
    # These values must be copied or referenced at the block level for local stencils.
    assert np.isclose(sample_block.rho, traceable_rho), (
        f"Traceability Breach: Density mismatch. Expected {traceable_rho}, got {sample_block.rho}"
    )
    assert np.isclose(sample_block.mu, traceable_mu), (
        f"Traceability Breach: Viscosity mismatch. Expected {traceable_mu}, got {sample_block.mu}"
    )

def test_gate_2a_foundation_mismatch_catch(monkeypatch):
    """
    Verification: Catch RuntimeError in src/step2/stencil_assembler.py 
    when the fields buffer width does not align with the FI Schema.
    Compliance: Rule 1 (Field Precision/Scale Audit) - Foundation Guard.
    """

    # 1. Setup: Valid state
    state = create_dummy_state(nx=2, ny=2, nz=2)
    
    # 2. Action: Force a mismatch in the buffer width
    # If the schema requires 10 fields, we provide 9.
    required_width = FI.num_fields()
    bad_width = required_width - 1
    
    # Create a corrupted buffer shape (Rule 9: N+2 padded volume)
    # Total cells: (2+2)^3 = 64
    corrupted_data = np.zeros((64, bad_width))
    
    # Monkeypatch the data attribute of the Physical Context (FieldManager)
    monkeypatch.setattr(state.fields, "data", corrupted_data)
    
    # 3. Verification: The Firewall must detect the drift from the FI Schema
    expected_error = f"Foundation Mismatch: Buffer width {bad_width} != Schema requirement {required_width}"
    
    with pytest.raises(RuntimeError, match=expected_error):
        assemble_stencil_matrix(state)

def test_gate_2a_registry_is_read_only():
    """
    Verification: Ensure the assembly process does not mutate the 
    Input Data / Geometric Context (Rule 4).
    """
    state = create_dummy_state(nx=2, ny=2, nz=2)
    original_nx = state.grid.nx
    
    _ = assemble_stencil_matrix(state)
    
    assert state.grid.nx == original_nx, "Rule 4 Violation: Assembly mutated the SSoT Grid object."