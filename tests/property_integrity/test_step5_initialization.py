# tests/property_integrity/test_step5_initialization.py

import numpy as np
import pytest

from src.common.simulation_context import SimulationContext
from src.common.solver_config import SolverConfig
from src.common.solver_state import (
    FieldManager,
    SimulationParameterManager,
    SolverState,
    GridManager,
    MaskManager,
    ManifestManager,
    DomainManager
)
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
        
        # Rule 5: Deterministic Init
        state = SolverState()
        state.iteration = 0 
        state.time = 0.0
        
        # Property assignment for ValidatedContainer managers
        params_manager = SimulationParameterManager()
        params_manager.time_step = input_data.simulation_parameters.time_step
        params_manager.total_time = input_data.simulation_parameters.total_time
        params_manager.output_interval = input_data.simulation_parameters.output_interval
        state.sim_params = params_manager
        
        # Rule 9: Initialize and allocate the contiguous Foundation
        fields = FieldManager()
        fields.allocate(n_cells=64) 
        state.fields = fields
        
        # Instantiate real managers to satisfy type validation
        grid = GridManager()
        grid.nx, grid.ny, grid.nz = 4, 4, 4
        grid.x_min, grid.x_max = 0.0, 1.0
        grid.y_min, grid.y_max = 0.0, 1.0
        grid.z_min, grid.z_max = 0.0, 1.0
        state.grid = grid
        
        # Initialize production-compliant managers
        masks = MaskManager()
        masks.mask = np.zeros((4, 4, 4))
        state.masks = masks
        
        manifest = ManifestManager()
        manifest.saved_snapshots = []
        state.manifest = manifest
        
        domain = DomainManager()
        domain.case_name = 'test_case'
        state.domain = domain
        
        return state, context

    def test_archivist_orchestration_contract(self, setup_state):
        """Rule 4: Verify Archivist receives valid configuration context."""
        state, context = setup_state
        state.iteration = 10 
        
        orchestrate_step5(state, context)
        
        assert len(state.manifest.saved_snapshots) > 0, "Snapshot must be recorded in manifest."

    def test_archival_decision_logic(self, setup_state):
        """Rule 5: Verify archival threshold is strictly iteration-dependent."""
        state, context = setup_state
        
        state.iteration = 10
        orchestrate_step5(state, context)
        
        assert any("snapshot_0010.h5" in s for s in state.manifest.saved_snapshots)