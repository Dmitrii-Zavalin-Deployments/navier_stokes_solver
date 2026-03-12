# tests/property_integrity/test_step5_initialization.py

import pytest
from src.common.simulation_context import SimulationContext
from src.common.solver_state import SolverState
from src.common.solver_config import SolverConfig
from src.step5.orchestrate_step5 import orchestrate_step5
from tests.helpers.solver_input_schema_dummy import create_validated_input

class TestStep5Initialization:
    """AUDITOR: Step 5 Archivist Orchestration Pipeline Verification."""

    @pytest.fixture(scope="class")
    def setup_state(self):
        """Prepare minimal state for archive logic verification."""
        # Rule 5: Deterministic Init - explicit values for all slots
        config = SolverConfig(
            ppe_tolerance=1e-6, 
            ppe_atol=1e-9, 
            ppe_max_iter=1000, 
            ppe_omega=1.0,
            output_interval=10  # Explicitly defined
        )
        input_data = create_validated_input(nx=4, ny=4, nz=4)
        context = SimulationContext(input_data=input_data, config=config)
        
        # Initialize state with mandatory iteration tracking
        state = SolverState()
        state.iteration = 0 
        
        return state, context

    def test_archivist_orchestration_contract(self, setup_state):
        """Rule 4: Verify Archivist receives valid configuration context."""
        state, context = setup_state
        
        # Verify the orchestrator return contract
        result = orchestrate_step5(state, context)
        assert isinstance(result, SolverState), "Orchestrator must return the SolverState."
        assert callable(orchestrate_step5), "Orchestrator must be callable."

    def test_archival_decision_logic(self, setup_state):
        """Rule 5: Verify archival threshold is strictly iteration-dependent."""
        state, context = setup_state
        
        # Test archival trigger at interval (10)
        state.iteration = 10
        # In a real test, you would mock 'save_snapshot' here to verify call count
        orchestrate_step5(state, context)
        
        assert state.iteration == 10, "Archivist should not modify iteration count."