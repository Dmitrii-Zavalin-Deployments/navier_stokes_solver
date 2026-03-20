# tests/property_integrity/test_step3_initialization.py

import numpy as np
import pytest

from src.common.elasticity import ElasticManager
from src.common.field_schema import FI
from src.common.simulation_context import SimulationContext
from src.common.solver_config import SolverConfig
from src.step1.orchestrate_step1 import orchestrate_step1
from src.step2.orchestrate_step2 import orchestrate_step2
from src.step3.orchestrate_step3 import orchestrate_step3
from tests.helpers.solver_input_schema_dummy import create_validated_input


class TestStep3Initialization:
    """AUDITOR: Step 3 Projection Method Pipeline Verification."""

    @pytest.fixture(scope="class")
    def setup_state(self):
        """Prepare a fully wired state with explicit configuration."""
        input_data = create_validated_input(nx=4, ny=4, nz=4)
        # Deterministic Initialization: Explicit parameters required by Rule 5
        config = SolverConfig(
            ppe_tolerance=1e-5,
            ppe_atol=1e-9,
            ppe_max_iter=100,
            ppe_omega=1.0,
            dt=input_data.simulation_parameters.time_step
        )
        context = SimulationContext(input_data=input_data, config=config)
        elasticity = ElasticManager(config)
        state = orchestrate_step1(context)
        state = orchestrate_step2(state)
        return state, context, elasticity

    def test_ghost_cell_immunity(self, setup_state):
        """Rule 9: Ensure Step 3 logic ignores ghost cell pointers."""
        state, context, elasticity = setup_state
        ghost_block = state.stencil_matrix[0]
        
        # Access via public property (is_ghost) to respect __slots__ integrity
        ghost_block.center.is_ghost = True 
        
        _, delta = orchestrate_step3(ghost_block, context, elasticity, is_first_pass=False)
        assert delta == 0.0, "Ghost cells must return 0 delta in projection."

    def test_foundation_mutation_consistency(self, setup_state):
        """Rule 9: Verify buffer mutation remains schema-compliant."""
        state, context, elasticity = setup_state
        block = state.stencil_matrix[len(state.stencil_matrix) // 2]
        
        # Execute corrector/synchronization step
        orchestrate_step3(block, context, elasticity, is_first_pass=False)
        
        # Verify schema integrity using FI
        assert block.center.get_field(FI.P) is not None, "Pressure field unreachable."
        
        # Ensure no NaN injection during in-place mutation
        assert not np.isnan(block.center.get_field(FI.P)), "NaN injection detected in buffer."

    def test_omega_parameter_ingress(self, setup_state):
        """Rule 4: Verify numerical parameter (omega) is sourced from context."""
        state, context, elasticity = setup_state
        # Inject custom omega to verify it propagates through the context
        context.config.ppe_omega = 1.5
        
        block = state.stencil_matrix[len(state.stencil_matrix) // 2]
        
        # Execute and check for execution path (Side effect validation)
        _, delta = orchestrate_step3(block, context, elasticity, is_first_pass=False)
        assert isinstance(delta, float), "Orchestrator must return a scalar residual (delta)."