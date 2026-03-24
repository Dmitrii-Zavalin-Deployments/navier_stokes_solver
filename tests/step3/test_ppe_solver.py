# tests/step3/test_ppe_solver.py

from unittest.mock import MagicMock, patch

import pytest

from src.step3.ppe_solver import solve_pressure_poisson_step


class TestPPESolverIntegrity:
    
    def test_catch_poisoned_pressure_slingshot(self, caplog):
        """
        Verify that a massive p_old (from a failed dt attempt) 
        triggers a Rule 7 ArithmeticError and logs the error.
        """
        # Setup mock block
        block = MagicMock()
        block.id = "test-block-01"
        block.dt = 1e-6
        block.dx = block.dy = block.dz = 0.1
        
        # Define the explicit threshold for this test
        test_threshold = 1.0e10
        
        # Neighbors MUST return 0.0 to avoid poisoning the sum
        for neighbor in [block.i_plus, block.i_minus, block.j_plus, block.j_minus, block.k_plus, block.k_minus]:
            neighbor.get_field.return_value = 0.0
            
        # The "slingshot" value: 1.4e13 > 1.0e10
        block.center.get_field.return_value = 1.4e13 
        
        # EXECUTION: Passing the threshold explicitly (New Signature)
        with pytest.raises(ArithmeticError) as excinfo:
            solve_pressure_poisson_step(block, divergence_threshold=test_threshold, omega=1.0)
            
        # VALIDATION: Verify Rule 7 Compliance
        assert "Poisoned Pressure Trial" in str(excinfo.value)
        assert "PPE CRITICAL" in caplog.text
        assert "Poisoned p_old" in caplog.text
        assert "Limit: 1.0e+10" in caplog.text

    def test_catch_nan_divergence(self, caplog):
        """
        Verify that if div_v_star is NaN, we raise ArithmeticError.
        Rule 7: Atomic Numerical Truth.
        """
        block = MagicMock()
        block.dx = block.dy = block.dz = 0.1
        block.center.get_field.return_value = 0.0 
        
        # Mock neighbor values to be safe
        for neighbor in [block.i_plus, block.i_minus, block.j_plus, block.j_minus, block.k_plus, block.k_minus]:
            neighbor.get_field.return_value = 0.0
        
        # Use patch to inject NaN divergence
        with patch("src.step3.ppe_solver.compute_local_divergence_v_star", return_value=float("nan")):
            with pytest.raises(ArithmeticError):
                # Signature update: (block, threshold, omega)
                solve_pressure_poisson_step(block, divergence_threshold=1e12, omega=1.0)
        
        assert "PPE MATH ERROR" in caplog.text

    def test_successful_ppe_update(self):
        """Verify normal operation returns a valid delta."""
        block = MagicMock()
        block.dx = block.dy = block.dz = 1.0
        block.dt = 0.01
        block.rho = 1.0
        
        # Valid physical values
        block.center.get_field.return_value = 0.0
        for neighbor in [block.i_plus, block.i_minus, block.j_plus, block.j_minus, block.k_plus, block.k_minus]:
            neighbor.get_field.return_value = 0.0
        
        # EXECUTION: Standard valid run
        delta = solve_pressure_poisson_step(block, divergence_threshold=1e12, omega=1.0)
        
        assert isinstance(delta, float)
        assert delta >= 0.0