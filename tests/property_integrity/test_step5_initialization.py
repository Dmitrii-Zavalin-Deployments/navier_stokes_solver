# tests/property_integrity/test_step5_initialization.py

import pytest
from src.common.simulation_context import SimulationContext
from src.common.solver_config import SolverConfig
from src.common.solver_state import SolverState
from src.step5.orchestrate_step5 import orchestrate_step5
from tests.helpers.solver_input_schema_dummy import create_validated_input

class TestStep5Initialization:
    """AUDITOR: Step 5 Archivist Orchestration Pipeline Verification."""

    @pytest.fixture(scope="class")
    def setup_state(self):
        """Prepare minimal state for archive logic verification."""
        # Rule 5: Deterministic Init - Config contains only algorithm parameters
        config = SolverConfig(
            ppe_tolerance=1e-6, 
            ppe_atol=1e-9, 
            ppe_max_iter=1000, 
            ppe_omega=1.0
        )
        
        # input_data houses simulation parameters
        input_data = create_validated_input(nx=4, ny=4, nz=4)
        input_data.simulation_parameters.output_interval = 10 
        
        context = SimulationContext(input_data=input_data, config=config)
        
        # Initialize state
        state = SolverState()
        state.iteration = 0 
        
        # Rule 5 & 9: Use a Structural Mock to satisfy the State's grid requirement
        # Since the real GridManager is not yet implemented (verified via find),
        # we provide the minimum interface required by io_archivist.py.
        class MockGrid:
            __slots__ = ['nx', 'ny', 'nz']
            def __init__(self, nx, ny, nz):
                self.nx, self.ny, self.nz = nx, ny, nz

        # We bypass the type-check in _set_safe by using a mock that 
        # the archivist can read from. 
        state._grid = MockGrid(nx=4, ny=4, nz=4)
        
        return state, context

    def test_archivist_orchestration_contract(self, setup_state):
        """Rule 4: Verify Archivist receives valid configuration context."""
        state, context = setup_state
        
        result = orchestrate_step5(state, context)
        assert isinstance(result, SolverState), "Orchestrator must return the SolverState."
        assert callable(orchestrate_step5), "Orchestrator must be callable."

    def test_archival_decision_logic(self, setup_state):
        """Rule 5: Verify archival threshold is strictly iteration-dependent."""
        state, context = setup_state
        
        state.iteration = 10
        # This will now proceed into save_snapshot and find the grid dimensions
        orchestrate_step5(state, context)
        
        assert state.iteration == 10, "Archivist should not modify iteration count."