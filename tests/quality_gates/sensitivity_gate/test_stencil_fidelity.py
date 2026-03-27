# tests/quality_gates/sensitivity_gate/test_stencil_fidelity.py

import pytest
from src.step2.stencil_assembler import assemble_stencil_matrix

def test_gate_2a_registry_traceability(solver_state_dummy):
    """
    Gate 2.A: Stencil Registry Traceability Audit
    Verification: Physical constants (rho, mu) must propagate from SolverState 
    to StencilBlock objects during Step 2 assembly.
    Compliance: Physical Logic Firewall - Deterministic Initialization.
    """
    # 1. Setup unique 'traceable' physical constants in the dummy state
    # These values ensure we aren't just passing zero/default values.
    traceable_rho = 13.5
    traceable_mu = 0.0017
    
    solver_state_dummy.fluid_properties.density = traceable_rho
    solver_state_dummy.fluid_properties.viscosity = traceable_mu
    
    # 2. Execute Step 2: Assembly
    # This triggers the CellRegistry and physics_params extraction
    stencils = assemble_stencil_matrix(solver_state_dummy)
    
    # 3. Verification: Ensure the Registry successfully wired the physics
    assert len(stencils) > 0, "Gate Breach: No stencils assembled."
    
    # Audit the first Core StencilBlock [0,0,0]
    sample_block = stencils[0]
    
    # Check physical constant propagation (Rule 5 compliance)
    assert sample_block.rho == traceable_rho, (
        f"Traceability Breach: Density mismatch. Expected {traceable_rho}, got {sample_block.rho}"
    )
    assert sample_block.mu == traceable_mu, (
        f"Traceability Breach: Viscosity mismatch. Expected {traceable_mu}, got {sample_block.mu}"
    )

def test_gate_2a_foundation_mismatch_catch(solver_state_dummy, monkeypatch):
    """
    Verification: Catch RuntimeError in src/step2/stencil_assembler.py 
    when the fields buffer width does not align with the FI Schema.
    """
    import numpy as np
    from src.common.field_schema import FI
    
    # Force a mismatch in the buffer width (e.g., 5 fields instead of Schema requirement)
    bad_width = FI.num_fields() - 1
    bad_data = np.zeros((10, 10, 10, bad_width))
    
    # Use monkeypatch to simulate a corrupted state buffer
    monkeypatch.setattr(solver_state_dummy.fields, "data", bad_data)
    
    expected_error = f"Foundation Mismatch: Buffer width {bad_width} != Schema requirement {FI.num_fields()}"
    
    with pytest.raises(RuntimeError, match=expected_error):
        assemble_stencil_matrix(solver_state_dummy)