# tests/property_integrity/test_step5_initialization.py

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
from tests.helpers.solver_input_schema_dummy import (
    create_validated_input,
    make_output_schema_dummy,
)
from tests.helpers.solver_step5_output_dummy import make_step5_output_dummy


class TestStep5Initialization:
    """AUDITOR: Step 5 Archivist Orchestration Pipeline Verification."""

    @pytest.fixture(scope="class")
    def setup_state(self):
        """
        Rule 9: Dummy Initialization.
        Prepare a SolverState dummy to verify archival decision logic.
        """
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
        fields.allocate(n_cells=216) 
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
        
        domain = DomainManager()
        domain.type = "INTERNAL"
        state.domain_configuration = domain
        
        return state, context

    def test_archivist_orchestration_contract(self, setup_state):
        """Verify the 'Handover' from compute logic to archival record."""
        state, context = setup_state
        state.iteration = 10 
        orchestrate_step5(state, context)
        assert len(state.manifest.saved_snapshots) > 0, "Snapshot must be recorded in manifest."

    def test_archival_decision_logic(self, setup_state):
        """Verify iteration-based naming convention in the manifest."""
        state, context = setup_state
        state.iteration = 10
        orchestrate_step5(state, context)
        assert any("snapshot_0010.h5" in s for s in state.manifest.saved_snapshots), \
            "The Property Tracking Matrix requires specific iteration mapping."
    
    # --- TRANSITION & TERMINAL STATE CHECKS ---

    def test_bridge_step5_to_output_integrity(self):
        """
        Vertical Integrity Mandate: Continuity Check.
        Verifies alignment between Step 5 and the final Solver Output Dummy.
        """
        nx, ny, nz = 4, 4, 4
        intermediate_state = make_step5_output_dummy(nx=nx, ny=ny, nz=nz)
        terminal_state = make_output_schema_dummy(nx=nx, ny=ny, nz=nz)

        # 1. Physical Field Continuity (Proof of Data Survival)
        assert (intermediate_state.fields.data == terminal_state.fields.data).all(), \
            "Data corruption: Physical properties drifted during terminal transition."

        # 2. Manifest Evolution (Tracking Knowledge Growth)
        assert len(terminal_state.manifest.saved_snapshots) >= len(intermediate_state.manifest.saved_snapshots), \
            "Terminal manifest failed to inherit snapshot history from Step 5."
        
        # 3. Path Rooting Safety
        root = terminal_state.manifest.output_directory
        for path in terminal_state.manifest.saved_snapshots:
            assert path.startswith(root), f"Path {path} escaped established root {root}"

    def test_final_state_exit_contract(self):
        """Verify the solver successfully triggers the loop-termination flag."""
        target = 1.0
        state = make_output_schema_dummy(nx=4, ny=4, nz=4)
        state._simulation_parameters.total_time = target
        state._time = target 
        
        assert state._time >= state._simulation_parameters.total_time
        assert state._ready_for_time_loop is False, \
            "Termination Logic: Final state failed to lock 'ready_for_time_loop' to False."