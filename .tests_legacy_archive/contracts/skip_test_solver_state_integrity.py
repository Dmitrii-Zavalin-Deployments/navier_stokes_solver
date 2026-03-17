# tests/contracts/test_solver_state_integrity.py

import numpy as np
import pytest

from src.solver_state import Diagnostics, FieldData, SolverState


class TestSolverStateIntegrity:
    """
    AUDIT GATE: Verifies that SolverState and its sub-containers satisfy 
    the project's structural contract and achieve 100% logic coverage.
    """

    def test_solver_state_facades_and_shortcuts(self):
        """Hits the property shortcuts (dt, rho, mu, inv_dx, etc.)."""
        state = SolverState()
        
        # Setup underlying data required for facades
        fluid_props = state.fluid = {"density": 1000.0, "viscosity": 0.001}
        state.config.simulation_parameters = {
            "time_step": 0.1, 
            "total_time": 1.0, 
            "output_interval": 1
        }
        state.grid.x_min, state.grid.x_max, state.grid.nx = 0.0, 1.0, 10
        state.grid.y_min, state.grid.y_max, state.grid.ny = 0.0, 1.0, 10
        state.grid.z_min, state.grid.z_max, state.grid.nz = 0.0, 1.0, 10
        
        # Physics Facades
        assert state.rho == 1000.0
        assert state.mu == 0.001
        
        # Numerical Facades
        assert state.dt == 0.1
        assert state.inv_dx == 10.0
        assert state.inv_dy == 10.0
        assert state.inv_dz == 10.0

    def test_field_data_dictionary_interface(self):
        """Hits lines 311-315: __getitem__ and __setitem__ logic."""
        fields = FieldData()
        arr = np.array([1.0, 2.0, 3.0])
        
        # Test Case-Insensitive set/get
        fields["U_STAR"] = arr
        assert np.array_equal(fields["u_star"], arr)
        
        # Test KeyError path (Line 315)
        with pytest.raises(KeyError, match="is not recognized"):
            _ = fields["INVALID_KEY"]
            
        with pytest.raises(KeyError, match="is not recognized"):
            fields["INVALID_KEY"] = arr

    def test_diagnostics_safe_get_and_mapping(self):
        """Hits lines 647, 656: get() and legacy __getitem__ mapping."""
        diag = Diagnostics()
        diag.memory_footprint_gb = 5.0
        
        # Test Legacy Key Mapping
        assert diag["memory_footprint"] == 5.0
        # Test safe get with default
        assert diag.get("missing", "default") == "default"

    def test_simulation_health_mapping(self):
        """Hits the health.get() mapping branches."""
        state = SolverState()
        state.health.max_u = 10.5
        assert state.health.get("max_velocity_magnitude") == 10.5
        assert state.health.get("none_key", 0.0) == 0.0

    def test_legacy_support_and_roundtrip(self):
        """
        Hits lines 764, 767, 780-795. 
        Initializes fields to avoid the 'Access Error' RuntimeError.
        """
        state = SolverState()
        dummy_arr = np.zeros((2, 2, 2))
        
        # Initialize everything to_legacy_dict needs
        state.fields.U_ext = dummy_arr
        state.fields.V_ext = dummy_arr
        state.fields.W_ext = dummy_arr
        state.fields.P_ext = dummy_arr
        
        # Test Legacy Export (Line 764 & 767)
        legacy = state.to_legacy_dict()
        assert legacy["ready_for_time_loop"] is False
        assert isinstance(legacy["step4_diagnostics"], Diagnostics)
        
        # Test JSON Safe (The full state dump)
        # Note: This will hit all to_dict() calls for sub-containers
        safe_json = state.to_json_safe()
        assert safe_json["iteration"] == 0
        assert "config" in safe_json