# tests/quality_gates/logic_gate/test_pressure_configuration.py

import pytest

from src.common.simulation_context import SimulationContext
from src.step1.orchestrate_step1 import orchestrate_step1
from tests.helpers.solver_input_schema_dummy import get_explicit_solver_config

# Rule 5: Explicit Mock Configuration
MOCK_CONFIG = {
    "ppe_tolerance": 1e-6,
    "ppe_atol": 1e-10,
    "ppe_max_iter": 100,
    "ppe_omega": 1.5,
    "dt_min_limit": 1e-6,
    "ppe_max_retries": 3,
    "divergence_threshold": 1e-4
}

class TestPressureBoundaryLogic:
    """
    Quality Gate: Validates the 'Reference Pressure' requirement.
    A valid simulation must have exactly one pressure reference point 
    to avoid being over-determined or floating.
    """

    def _create_context_with_bcs(self, bc_list):
        """Helper to inject specific BC combinations into a context."""
        input_dict = get_explicit_solver_config(nx=4, ny=4, nz=4)
        input_dict["boundary_conditions"] = bc_list
        # Rule 5: Deterministic Context Creation
        return SimulationContext.create(input_dict, MOCK_CONFIG.copy())

    def test_pass_pressure_at_outflow_only(self):
        """Standard Case: P fixed at Outflow, V fixed at Inflow."""
        bcs = [
            {"location": "x_min", "type": "inflow", "values": {"u": 1.0, "v": 0.0, "w": 0.0}},
            {"location": "x_max", "type": "outflow", "values": {"p": 0.0}}
        ]
        context = self._create_context_with_bcs(bcs)
        state = orchestrate_step1(context)
        
        # Verify that p is present in the outflow BC manager
        outflow = next(bc for bc in state.boundary_conditions.items if bc.location == "x_max")
        assert "p" in outflow.values
        assert "u" not in outflow.values

    def test_pass_pressure_at_inflow_only(self):
        """Reverse Case: P fixed at Inflow, Outflow is zero-gradient (floating)."""
        bcs = [
            {"location": "x_min", "type": "inflow", "values": {"p": 101325.0}},
            {"location": "x_max", "type": "outflow", "values": {"u": 1.0, "v": 0.0, "w": 0.0}}
        ]
        context = self._create_context_with_bcs(bcs)
        state = orchestrate_step1(context)
        
        inflow = next(bc for bc in state.boundary_conditions.items if bc.location == "x_min")
        assert "p" in inflow.values
        print("✅ Passed: System has a single reference point at Inflow.")

    def test_fail_dual_pressure_reference(self):
        """
        Failure Case: Over-determined system.
        Fixing pressure at both ends conflicts with the internal 
        continuity calculations in Step 3.
        """
        bcs = [
            {"location": "x_min", "type": "inflow", "values": {"u": 1.0, "p": 100.0}},
            {"location": "x_max", "type": "outflow", "values": {"p": 0.0}}
        ]
        
        # Depending on your implementation, this should raise a ValueError 
        # during SimulationContext validation or Step 1 Orchestration.
        with pytest.raises(ValueError, match="Over-determined pressure"):
            context = self._create_context_with_bcs(bcs)
            orchestrate_step1(context)

    def test_fail_no_pressure_reference(self):
        """
        Failure Case: Under-determined (Floating) system.
        If no 'p' is set anywhere, the Poisson solver will drift to infinity.
        """
        bcs = [
            {"location": "x_min", "type": "inflow", "values": {"u": 1.0, "v": 0.0, "w": 0.0}},
            {"location": "x_max", "type": "outflow", "values": {"u": 1.0, "v": 0.0, "w": 0.0}}
        ]
        
        with pytest.raises(ValueError, match="No pressure reference found"):
            context = self._create_context_with_bcs(bcs)
            orchestrate_step1(context)