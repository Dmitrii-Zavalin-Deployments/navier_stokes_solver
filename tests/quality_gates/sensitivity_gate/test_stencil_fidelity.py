# tests/quality_gates/sensitivity_gate/test_stencil_fidelity.py

import numpy as np
import pytest

from src.common.field_schema import FI
from src.step1.orchestrate_step1 import orchestrate_step1
from src.step2.stencil_assembler import assemble_stencil_matrix
from tests.helpers.solver_input_schema_dummy import create_validated_input


def test_gate_2a_registry_traceability():
    """
    Gate 2.A: Stencil Registry Traceability Audit
    Verification: Physical constants propagate from SolverState to StencilBlocks.
    Compliance: Rule 4 (SSoT) & Rule 5 (Deterministic Initialization).
    """

    # 1. Setup: Use validated input to build the context
    nx, ny, nz = 2, 2, 2
    context = create_validated_input(nx=nx, ny=ny, nz=nz)

    # Setup unique 'traceable' physical constants (Rule 7: Atomic Truth)
    traceable_rho = 13.52
    traceable_mu = 0.00173
    
    context.input_data.fluid_properties.density = traceable_rho
    context.input_data.fluid_properties.viscosity = traceable_mu
    
    # 2. Action: Hydrate the state through Step 1
    # This bypasses the broken solver_state_dummy dependency
    state = orchestrate_step1(context)
    
    # 3. Execute Step 2 Assembly
    stencils = assemble_stencil_matrix(state)
    
    # 4. Verification: Registry successfully wired the physics
    assert len(stencils) == (nx * ny * nz), "Gate Breach: Stencil count mismatch."
    
    sample_block = stencils[0]
    
    # Values must be correctly mirrored in the block level for local stencils
    assert np.isclose(sample_block.rho, traceable_rho), (
        f"Traceability Breach: Density mismatch. Expected {traceable_rho}, got {sample_block.rho}"
    )
    assert np.isclose(sample_block.mu, traceable_mu), (
        f"Traceability Breach: Viscosity mismatch. Expected {traceable_mu}, got {sample_block.mu}"
    )

def test_gate_2a_foundation_mismatch_catch(monkeypatch):
    """
    Verification: Catch RuntimeError when fields buffer width deviates from Schema.
    Compliance: Rule 1 (Field Precision/Scale Audit) - Foundation Guard.
    """

    # 1. Setup: Valid state hydration
    context = create_validated_input(nx=2, ny=2, nz=2)
    state = orchestrate_step1(context)
    
    # 2. Action: Force a mismatch in the buffer width
    required_width = FI.num_fields()
    bad_width = required_width - 1
    
    # Corrupt the Foundation (Rule 9: Padded volume = 4^3 = 64)
    corrupted_data = np.zeros((64, bad_width))
    
    # Monkeypatch the data attribute of the FieldManager
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
    # 1. Setup: Hydrate valid state
    context = create_validated_input(nx=2, ny=2, nz=2)
    state = orchestrate_step1(context)
    
    original_nx = state.grid.nx
    
    # 2. Action: Run assembly
    _ = assemble_stencil_matrix(state)
    
    # 3. Verification: SSoT remains untouched
    assert state.grid.nx == original_nx, (
        "Rule 4 Violation: Assembly mutated the SSoT Grid object. "
        "Geometric Context must be read-only during logic hydration."
    )