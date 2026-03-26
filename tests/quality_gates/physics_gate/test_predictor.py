# tests/quality_gates/physics_gate/test_predictor.py

import logging

import numpy as np
import pytest

from src.common.field_schema import FI
from src.step3.predictor import compute_local_predictor_step
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


def setup_predictor_block(dt=1.0, rho=1.0, mu=1.0, dx=1.0):
    """
    Wires a StencilBlock with unit physics for analytical transparency.
    """
    state = make_step2_output_dummy(nx=4, ny=4, nz=4)
    block = state.stencil_matrix[10]  # Central core block

    # Standardize physics to unit values for clean math verification
    params = {
        '_dx': float(dx), '_dy': float(dx), '_dz': float(dx),
        '_dt': float(dt), '_rho': float(rho), '_mu': float(mu),
        '_f_vals': (0.0, 0.0, 0.0) 
    }
    for attr, val in params.items():
        object.__setattr__(block, attr, val)

    # Clean the buffer to ensure no residual data contamination
    block.center.fields_buffer.fill(0.0) 
    return block

# --- PHYSICS INTEGRATION TESTS ---

def test_predictor_diffusion_only():
    """
    Verifies: v* = v_n + (dt/rho) * (mu * lap(v_n))
    Setup: lap=2.0, all other terms 0.
    Expectation: 0 + (1/1) * (1.0 * 2.0) = 2.0
    """
    block = setup_predictor_block(mu=1.0, dt=1.0, rho=1.0)
    
    # Curvature in X: (ip: 1.0, c: 0.0, im: 1.0) -> lap = 2.0
    block.i_plus.set_field(FI.VX, 1.0)
    block.i_minus.set_field(FI.VX, 1.0)
    
    compute_local_predictor_step(block)
    
    obtained = block.center.get_field(FI.VX_STAR)
    assert obtained == pytest.approx(2.0), f"Diffusion failed: expected 2.0, got {obtained}"

def test_predictor_full_3d_complex_integration():
    """
    Full Equation: v* = v_n + (dt/rho) * [ (mu * lap) - (rho * adv) + F - grad_p ]
    Setup: 
    - v_n = 1.0
    - lap = 0.0 (Linear slope)
    - adv = 1.0 (u * du/dx = 1 * 1)
    - grad_p = 1.0
    - F = 0.0
    Calculation: 1.0 + (1/1) * [ (1*0) - (1*1) + 0 - 1 ] = -1.0
    """
    block = setup_predictor_block(mu=1.0, dt=1.0, rho=1.0, dx=1.0)
    
    # 1. Velocity slope: (im:0, c:1, ip:2) -> grad_v=1, lap_v=0
    block.center.set_field(FI.VX, 1.0)
    block.i_plus.set_field(FI.VX, 2.0)
    block.i_minus.set_field(FI.VX, 0.0)
    
    # 2. Match Y/Z neighbors to center to keep Laplacian 1D
    for n in [block.j_plus, block.j_minus, block.k_plus, block.k_minus]:
        n.set_field(FI.VX, 1.0)
        
    # 3. Pressure gradient: (ip:2, im:0) -> grad_p = 1.0
    block.i_plus.set_field(FI.P, 2.0)
    block.i_minus.set_field(FI.P, 0.0)

    compute_local_predictor_step(block)
    
    obtained = block.center.get_field(FI.VX_STAR)
    assert obtained == pytest.approx(-1.0), f"Complex integration failed: expected -1.0, got {obtained}"

# --- FORENSIC LOGGING & AUDIT TESTS ---

def test_predictor_audit_logging(caplog):
    """Verifies AUDIT debug logs are generated for operator outputs."""
    block = setup_predictor_block()
    
    with caplog.at_level(logging.DEBUG):
        compute_local_predictor_step(block)
        
    assert "DEBUG [Predictor]" in caplog.text
    assert "PREDICT [Success]" in caplog.text
    assert "Type=" in caplog.text

def test_predictor_component_info_logging(caplog):
    """Verifies VX_STAR value is logged at INFO level."""
    block = setup_predictor_block()
    
    with caplog.at_level(logging.INFO):
        compute_local_predictor_step(block)
        
    assert "DEBUG [Predictor]" in caplog.text
    assert "VX_STAR:" in caplog.text

def test_predictor_contamination_recovery(caplog):
    """
    Verifies Rule 7: DNA Audit.
    Ensures that if the Laplacian promotes a float to an array, the 
    predictor logs the contamination and recovers using .
    """
    block = setup_predictor_block()
    
    # 1. Access the predictor module for monkeypatching
    from src.step3 import predictor
    original_lap = predictor.compute_local_laplacian_v_n
    
    # 2. Inject a 1-element array to trigger the hasattr(val, "__len__") check
    # We return a tuple where the first element is a NumPy array
    predictor.compute_local_laplacian_v_n = lambda b: (np.array([0.0]), 0.0, 0.0)
    
    try:
        # 3. FIX: Explicitly target the "Solver.Predictor" logger
        # This captures the ERROR even if propagation is disabled
        with caplog.at_level(logging.ERROR, logger="Solver.Predictor"):
            compute_local_predictor_step(block)
            
        # 4. Forensic Verification
        assert "CONTAMINATION DETECTED" in caplog.text
        # Verify it was logged as an ERROR, not just a warning
        assert any(record.levelname == "ERROR" for record in caplog.records)
        
    finally:
        # 5. Restore original function to maintain test isolation
        predictor.compute_local_laplacian_v_n = original_lap

def test_predictor_math_failure_traceback(caplog):
    """Verifies CRITICAL log on mathematical collapse (Rule 7)."""
    block = setup_predictor_block()
    
    # 1. Force a specific contract failure by deleting the viscosity coefficient
    delattr(block, '_mu')
    
    with caplog.at_level(logging.CRITICAL):
        # 2. Rule 7/B017: Target AttributeError specifically to ensure 
        # the test doesn't pass for the wrong reasons (like a SyntaxError).
        with pytest.raises(AttributeError):
            compute_local_predictor_step(block)
            
    # 3. Verify the forensic log was captured before the exception was bubbled
    assert "MATH FAILURE" in caplog.text