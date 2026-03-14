import numpy as np
import pytest

from src.common.simulation_context import SimulationContext
from src.common.solver_config import SolverConfig
from src.common.solver_state import (
    DomainManager,
    FieldManager,
    GridManager,
    ManifestManager,
    MaskManager,
    SimulationParameterManager,
    SolverState,
)
from src.step5.orchestrate_step5 import orchestrate_step5
from tests.helpers.solver_input_schema_dummy import create_validated_input


class TestStep5Initialization:
    """AUDITOR: Step 5 Archivist Orchestration Pipeline Verification."""

    @pytest.fixture(scope="class")
    def setup_state(self):
        """Prepare minimal state for archive logic verification."""
        config = SolverConfig(
            ppe_tolerance=1e-6, 
            ppe_atol=1e-9, 
            ppe_max_iter=1000, 
            ppe_omega=1.0
        )
        
        input_data = create_validated_input(nx=4, ny=4, nz=4)
        input_data.simulation_parameters.output_interval = 10 
        
        context = SimulationContext(input_data=input_data, config=config)
        
        state = SolverState()
        state.iteration = 0 
        state.time = 0.0
        
        params_manager = SimulationParameterManager()
        params_manager.time_step = input_data.simulation_parameters.time_step
        params_manager.total_time = input_data.simulation_parameters.total_time
        params_manager.output_interval = input_data.simulation_parameters.output_interval
        state.simulation_parameters = params_manager
        
        fields = FieldManager()
        fields.allocate(n_cells=64) 
        state.fields = fields
        
        grid = GridManager()
        grid.nx, grid.ny, grid.nz = 4, 4, 4
        grid.x_min, grid.x_max = 0.0, 1.0
        grid.y_min, grid.y_max = 0.0, 1.0
        grid.z_min, grid.z_max = 0.0, 1.0
        state.grid = grid
        
        masks = MaskManager()
        masks.mask = np.zeros((4, 4, 4))
        state.mask = masks
        
        manifest = ManifestManager()
        manifest.saved_snapshots = []
        state.manifest = manifest
        
        # DomainManager is now initialized normally as the production code expects.
        domain = DomainManager()
        domain.type = "INTERNAL"
        state.domain_configuration = domain
        
        return state, context

    def test_archivist_orchestration_contract(self, setup_state):
        state, context = setup_state
        state.iteration = 10 
        orchestrate_step5(state, context)
        assert len(state.manifest.saved_snapshots) > 0, "Snapshot must be recorded in manifest."

    def test_archival_decision_logic(self, setup_state):
        state, context = setup_state
        state.iteration = 10
        orchestrate_step5(state, context)
        assert any("snapshot_0010.h5" in s for s in state.manifest.saved_snapshots), "Missing snapshot in manifest."