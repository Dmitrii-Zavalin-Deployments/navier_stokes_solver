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
        
    assert "DEBUG [Predictor]: Type=Sovereign" in caplog.text
    assert "PREDICT [Success]" in caplog.text
    assert "Type=" in caplog.text

def test_predictor_component_info_logging(caplog):
    """Verifies VX_STAR value is logged at INFO level."""
    block = setup_predictor_block()
    
    with caplog.at_level(logging.DEBUG):
        compute_local_predictor_step(block)
        
    assert "DEBUG [Predictor]: Type=Sovereign" in caplog.text
    assert "VX_STAR:" in caplog.text

def test_predictor_contamination_recovery(caplog):
    """
    Verifies Rule 7: DNA Audit with Forensic Tracing.
    """
    print("\n--- [START] test_predictor_contamination_recovery ---")
    
    # 1. Setup Block
    block = setup_predictor_block()
    print(f"DEBUG [Test]: Block initialized | ID: {getattr(block, 'id', 'Unknown')}")
    
    # 2. Access the predictor module
    from src.step3 import predictor
    original_lap = predictor.compute_local_laplacian_v_n
    print("DEBUG [Test]: Captured original_lap function reference.")
    
    # 3. Inject the "Contaminant" (NumPy array instead of float)
    contaminant = (np.array([0.0]), 0.0, 0.0)
    predictor.compute_local_laplacian_v_n = lambda b: contaminant
    print(f"DEBUG [Test]: Monkeypatch injected | Value: {contaminant} | Type of [0]: {type(contaminant[0])}")
    
    try:
        print("DEBUG [Test]: Entering caplog context (Level: ERROR, Logger: Solver.Predictor)")
        with caplog.at_level(logging.ERROR, logger="Solver.Predictor"):
            
            print("DEBUG [Test]: Calling compute_local_predictor_step...")
            compute_local_predictor_step(block)
            print("DEBUG [Test]: compute_local_predictor_step returned successfully.")
            
        # 4. Forensic Verification of the Logs
        log_content = caplog.text
        print(f"DEBUG [Test]: Captured Log Content Length: {len(log_content)} chars")
        
        # Check for the specific failure string
        has_failure_msg = "PREDICTOR FAILURE" in log_content
        print(f"DEBUG [Test]: Search for 'PREDICTOR FAILURE' | Found: {has_failure_msg}")
        assert has_failure_msg
        
        # Verify it was logged as CRITICAL (Rule 3 Sync)
        critical_records = [r for r in caplog.records if r.levelname == "CRITICAL"]
        print(f"DEBUG [Test]: Critical Records Found: {len(critical_records)}")
        for i, rec in enumerate(critical_records):
            print(f"  -> Record {i}: {rec.message}")
            
        assert len(critical_records) > 0, "No CRITICAL level logs found in caplog.records"
        
    except Exception as e:
        print(f"ERROR [Test]: Unexpected Exception during execution: {type(e).__name__} - {e}")
        raise e
        
    finally:
        # 5. Restore original function
        predictor.compute_local_laplacian_v_n = original_lap
        print("DEBUG [Test]: Restored original_lap. Test isolation preserved.")
        print("--- [END] test_predictor_contamination_recovery ---\n")

def test_predictor_math_failure_traceback(caplog):
    """Verifies CRITICAL log on mathematical collapse (Rule 7) with Trace."""
    print("\n--- [START] test_predictor_math_failure_traceback ---")
    
    # 1. Setup Block and identify state
    block = setup_predictor_block()
    has_mu_initially = hasattr(block, '_mu')
    print(f"DEBUG [Test]: Block ID: {getattr(block, 'id', 'Unknown')} | Has _mu: {has_mu_initially}")
    
    # 2. Intentional Sabotage (Rule 5: Zero-Default Policy check)
    # Deleting _mu should trigger an immediate AttributeError in Rule-compliant code.
    print("DEBUG [Test]: Sabotaging block by deleting '_mu' attribute...")
    if has_mu_initially:
        delattr(block, '_mu')
    print(f"DEBUG [Test]: Attribute check after deletion | Has _mu: {has_mu_initially}")
    
    # 3. Execution Context
    print("DEBUG [Test]: Entering caplog context (Level: DEBUG)")
    with caplog.at_level(logging.DEBUG):
        
        print("DEBUG [Test]: Expecting AttributeError from compute_local_predictor_step...")
        try:
            with pytest.raises(AttributeError) as exc_info:
                compute_local_predictor_step(block)
            
            print(f"DEBUG [Test]: Successfully caught expected exception: {exc_info.type}")
            print(f"DEBUG [Test]: Exception message: {exc_info.value}")
            
        except Exception as e:
            print(f"ERROR [Test]: Caught WRONG exception type: {type(e).__name__} - {e}")
            raise e

    # 4. Forensic Log Audit
    log_text = caplog.text
    print(f"DEBUG [Test]: Captured Log Length: {len(log_text)} chars")
    
    # Search for the "Sovereign" marker (The 'Black Box' Recording)
    search_string = "DEBUG [Predictor]: Type=Sovereign"
    found_marker = search_string in log_text
    print(f"DEBUG [Test]: Searching for '{search_string}' | Found: {found_marker}")
    
    if not found_marker:
        print("DEBUG [Test]: DUMPING FULL LOG FOR ANALYSIS:")
        print("------------------------------------------")
        print(log_text if log_text else "[EMPTY LOG]")
        print("------------------------------------------")
    
    assert found_marker, "Forensic log missing! Code crashed before reaching Sovereign access."
    print("--- [END] test_predictor_math_failure_traceback ---\n")