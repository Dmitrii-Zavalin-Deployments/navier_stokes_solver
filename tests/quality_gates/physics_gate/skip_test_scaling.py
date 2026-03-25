# tests/quality_gates/physics_gate/test_scaling.py

import pytest

from src.step3.ops.scaling import get_dt_over_rho, get_rho_over_dt
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy


def setup_scaling_block(block, dt, rho):
    """
    Manually injects physical constants into the StencilBlock slots.
    Mimics Step 2 Assembly where state.config values are mapped to objects.
    """
    # Targets protected slots to ensure we are testing the SSoT interface
    object.__setattr__(block, '_dt', float(dt))
    object.__setattr__(block, '_rho', float(rho))
    return block

@pytest.mark.parametrize("dt, rho, expected_dt_rho, expected_rho_dt", [
    (0.01, 1.0, 0.01, 100.0),    # Standard Water-like case
    (0.001, 1000.0, 1e-6, 1e6), # High density / Small step
    (0.5, 0.5, 1.0, 1.0),       # Unit symmetry
    (1.0, 1.225, 1/1.225, 1.225) # Air-like density at sea level
])
def test_scaling_factors_accuracy(dt, rho, expected_dt_rho, expected_rho_dt):
    """
    Gate: Verifies scaling factors used in Predictor and Projection steps.
    Ensures Rule 4 (SSoT) and Rule 7 (Numerical Truth) compliance.
    """
    # 1. Arrange
    block = make_step3_output_dummy()
    setup_scaling_block(block, dt, rho)
    
    # 2. Act
    dt_over_rho = get_dt_over_rho(block)
    rho_over_dt = get_rho_over_dt(block)
    
    # 3. Assert (Rule 7: Machine Precision)
    assert dt_over_rho == pytest.approx(expected_dt_rho, abs=1e-15)
    assert rho_over_dt == pytest.approx(expected_rho_dt, abs=1e-15)

def test_scaling_division_by_zero_protection():
    """
    Gate: Ensures the system raises an error if dt or rho are uninitialized.
    Validates Rule 5 (Deterministic Initialization Mandate).
    """
    block = make_step3_output_dummy()
    
    # FORCE CORRUPTION: Overwrite the valid values from the dummy with None
    # This simulates a failure in the configuration/assembly pipeline.
    object.__setattr__(block, '_dt', None)
    object.__setattr__(block, '_rho', None)
    
    # Now it MUST raise a TypeError because you can't divide NoneType
    with pytest.raises((AttributeError, TypeError, ZeroDivisionError)):
        get_dt_over_rho(block)