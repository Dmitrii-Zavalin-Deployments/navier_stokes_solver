# tests/step3/test_ppe_solver.py

from unittest.mock import MagicMock

import pytest

from src.step3.ppe_solver import solve_pressure_poisson_step


class TestPPESolverIntegrity:
    
    def test_catch_poisoned_pressure_slingshot(self, caplog):
        """
        Verify that a massive p_old (from a failed dt attempt) 
        triggers a Rule 7 ArithmeticError and logs the error.
        """
        # Setup mock block with "poisoned" values
        block = MagicMock()
        block.id = "test-block-01"
        block.dt = 1e-6
        block.dx = block.dy = block.dz = 0.1
        
        # CRITICAL: Neighbors MUST return 0.0, not another Mock
        for neighbor in [block.i_plus, block.i_minus, block.j_plus, block.j_minus, block.k_plus, block.k_minus]:
            neighbor.get_field.return_value = 0.0
            
        block.center.get_field.return_value = 1.4e13 # The "slingshot" value
        
        # Ensure we capture the exception info (excinfo) to verify Rule 7 Compliance
        with pytest.raises(ArithmeticError) as excinfo:
            solve_pressure_poisson_step(block, omega=1.0)
            
        # Assertions: Verify the "Arithmetic Truth" of the failure
        assert "Poisoned Pressure Trial" in str(excinfo.value)
        assert "PPE CRITICAL" in caplog.text
        # Use a more robust check or ensure the format matches exactly
        assert "Value: 1.4000e+13" in caplog.text

    def test_catch_nan_divergence(self, caplog):
        """
        Verify that if div_v_star is NaN, we raise ArithmeticError 
        to trigger the rollback.
        """
        block = MagicMock()
        block.dx = block.dy = block.dz = 0.1
        block.center.get_field.return_value = 0.0 # Stable pressure
        
        # Force a math error in the RHS calculation
        with pytest.warns(RuntimeWarning): # Handle potential numpy warnings
             # Mock the div calc to return NaN
             import src.step3.ppe_solver as ppe
             ppe.compute_local_divergence_v_star = MagicMock(return_value=float('nan'))
             
             with pytest.raises(ArithmeticError):
                 solve_pressure_poisson_step(block, omega=1.0)
        
        assert "PPE MATH ERROR" in caplog.text

    def test_successful_ppe_update(self):
        """Verify normal operation returns a valid delta."""
        block = MagicMock()
        block.dx = block.dy = block.dz = 1.0
        block.dt = 0.01
        block.rho = 1.0
        
        # Valid physical values
        block.center.get_field.side_effect = lambda field: 0.0
        block.i_plus.get_field.return_value = 0.0
        block.i_minus.get_field.return_value = 0.0
        # ... and so on for other neighbors
        
        delta = solve_pressure_poisson_step(block, omega=1.0)
        assert isinstance(delta, float)
        assert delta >= 0.0