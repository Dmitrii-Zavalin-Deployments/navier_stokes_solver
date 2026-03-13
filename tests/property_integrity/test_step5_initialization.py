# tests/property_integrity/test_step5_initialization.py

import pytest
import numpy as np
from src.common.simulation_context import SimulationContext
from src.common.solver_config import SolverConfig
from src.common.solver_state import SolverState, FieldManager
from src.step5.orchestrate_step5 import orchestrate_step5
from tests.helpers.solver_input_schema_dummy import create_validated_input


class TestStep5Initialization:
    """AUDITOR: Step 5 Archivist Orchestration Pipeline Verification."""

    @pytest.fixture(scope="class")
    def setup_state(self):
        """Prepare minimal state for archive logic verification."""
        # Rule 5: Explicitly define config with no fallbacks
        config = SolverConfig(
            ppe_tolerance=1e-6, 
            ppe_atol=1e-9, 
            ppe_max_iter=1000, 
            ppe_omega=1.0
        )
        
        input_data = create_validated_input(nx=4, ny=4, nz=4)
        input_data.simulation_parameters.output_interval = 10 
        
        context = SimulationContext(input_data=input_data, config=config)
        
        # Rule 5: Deterministic Init (no defaults assumed)
        state = SolverState()
        state.iteration = 0 
        
        # Rule 9: Initialize and allocate the contiguous Foundation
        fields = FieldManager()
        fields.allocate(n_cells=64) 
        state.fields = fields
        
        # Rule 0: Mandatory __slots__ and Rule 9: Foundation-Object Bridge
        class MockGrid:
            __slots__ = [
                'nx', 'ny', 'nz', 'dx', 'dy', 'dz', 
                'x_mesh', 'y_mesh', 'z_mesh', 'mask_mesh'
            ]
            
            def __init__(self, nx, ny, nz):
                self.nx, self.ny, self.nz = nx, ny, nz
                self.dx = self.dy = self.dz = 0.1
                
                # Rule 9: Contiguous NumPy buffers for geometric fields
                shape = (nx, ny, nz)
                self.x_mesh = np.zeros(shape, dtype=np.float32)
                self.y_mesh = np.zeros(shape, dtype=np.float32)
                self.z_mesh = np.zeros(shape, dtype=np.float32)
                self.mask_mesh = np.zeros(shape, dtype=np.int32)

        # Rule 4: Hierarchy over Convenience
        state._grid = MockGrid(nx=4, ny=4, nz=4)
        
        return state, context

    def test_archivist_orchestration_contract(self, setup_state):
        """Rule 4: Verify Archivist receives valid configuration context."""
        state, context = setup_state
        state.iteration = 0 
        
        orchestrate_step5(state, context)
        
        # Rule 8 & 4: Accessing manifest via the new ManifestManager container
        assert len(state.manifest.saved_snapshots) > 0, "Snapshot must be recorded in manifest."

    def test_archival_decision_logic(self, setup_state):
        """Rule 5: Verify archival threshold is strictly iteration-dependent."""
        state, context = setup_state
        
        # Force iteration 10 to trigger snapshot based on output_interval=10
        state.iteration = 10
        orchestrate_step5(state, context)
        
        # Rule 8: Access via typed container property
        assert any("snapshot_0010.h5" in s for s in state.manifest.saved_snapshots)