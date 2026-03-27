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
    Compliance: Rule 4 (SSoT Hierarchy)
    """
    # 1. Setup: create_validated_input returns the top-level SolverInput
    nx, ny, nz = 2, 2, 2
    context = create_validated_input() 
    
    # Fix: Manually set grid dimensions in the SSoT container
    context.input_data.grid.nx = nx
    context.input_data.grid.ny = ny
    context.input_data.grid.nz = nz

    # Fix: Traceable constants must live in fluid_properties sub-container
    traceable_rho = 13.52
    traceable_mu = 0.00173
    context.input_data.fluid_properties.density = traceable_rho
    context.input_data.fluid_properties.viscosity = traceable_mu
    
    # 2. Action: Hydrate through the actual orchestrator
    state = orchestrate_step1(context)
    stencils = assemble_stencil_matrix(state)
    
    # 3. Verification
    sample_block = stencils[0]
    assert np.isclose(sample_block.rho, traceable_rho)
    assert np.isclose(sample_block.mu, traceable_mu)

def test_gate_2a_foundation_mismatch_catch(monkeypatch):
    """
    Verification: Catch RuntimeError when buffer width deviates from Schema.
    """
    context = create_validated_input()
    # Ensure dimensions match expected corrupted buffer size (Rule 9: 4^3 = 64)
    context.input_data.grid.nx = 2
    context.input_data.grid.ny = 2
    context.input_data.grid.nz = 2
    
    state = orchestrate_step1(context)
    
    required_width = FI.num_fields()
    bad_width = required_width - 1
    corrupted_data = np.zeros((64, bad_width))
    
    monkeypatch.setattr(state.fields, "data", corrupted_data)
    
    with pytest.raises(RuntimeError, match="Foundation Mismatch"):
        assemble_stencil_matrix(state)

def test_gate_2a_registry_is_read_only():
    """Rule 4: Geometric Context must be read-only during assembly."""
    context = create_validated_input()
    state = orchestrate_step1(context)
    original_nx = state.grid.nx
    
    _ = assemble_stencil_matrix(state)
    assert state.grid.nx == original_nx