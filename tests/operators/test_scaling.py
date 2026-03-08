# tests/operators/test_scaling.py

import pytest

from src.step3.ops.scaling import get_dt_over_rho, get_rho_over_dt


def test_scaling_factors():
    dt = 0.01
    rho = 1.225
    
    # Audit Proof
    expected_dt_rho = 0.01 / 1.225
    expected_rho_dt = 1.225 / 0.01
    
    # Verification
    assert get_dt_over_rho(dt, rho) == expected_dt_rho
    assert get_rho_over_dt(dt, rho) == expected_rho_dt

def test_scaling_zero_division():
    # Ensure the code fails gracefully if rho is zero
    with pytest.raises(ZeroDivisionError):
        get_dt_over_rho(0.01, 0)