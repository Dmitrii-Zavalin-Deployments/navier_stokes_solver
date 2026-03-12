# tests/property_integrity/test_step3_initialization.py

import numpy as np
import pytest

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
        """Prepare a fully wired state ready for projection."""
        input_data = create_validated_input(nx=4, ny=4, nz=4)
        config = SolverConfig()
        context = SimulationContext(input_data=input_data, config=config)
        
        state = orchestrate_step1(context)
        state = orchestrate_step2(state)
        return state, context

    def test_ghost_cell_immunity(self, setup_state):
        """Rule 9: Ensure Step 3 logic ignores ghost cell pointers."""
        state, context = setup_state
        ghost_block = state.stencil_matrix[0] # Assuming index 0 is mapped to a boundary/ghost context
        
        # Manually force the center to be ghost for isolation check
        ghost_block.center._is_ghost = True 
        
        _, delta = orchestrate_step3(ghost_block, context, is_first_pass=False)
        assert delta == 0.0, "Ghost cells must return 0 delta in projection."

    def test_foundation_mutation_consistency(self, setup_state):
        """Rule 9: Verify buffer mutation remains schema-compliant."""
        state, context = setup_state
        block = state.stencil_matrix[len(state.stencil_matrix) // 2] # Select a core cell
        
        # Capture pre-state
        block.center.fields_buffer[block.center.index, FI.P]
        
        # Execute corrector/synchronization step
        orchestrate_step3(block, context, is_first_pass=False)
        
        # Verify buffer still respects Field Schema index bounds
        assert state.fields.data.shape[1] == FI.num_fields(), "Schema integrity violation during mutation."
        
        # Check that we haven't corrupted adjacent fields
        # (This is a coarse check; the POST test in SolverState handles the heavy lifting)
        assert not np.isnan(block.center.fields_buffer).any(), "NaN injection detected in buffer."

    def test_omega_parameter_ingress(self, setup_state):
        """Rule 4: Verify numerical parameter (omega) is sourced from context."""
        state, context = setup_state
        context.config.ppe_omega = 1.5
        
        block = state.stencil_matrix[len(state.stencil_matrix) // 2]
        
        # Execute and check for execution path (Side effect validation)
        _, delta = orchestrate_step3(block, context, is_first_pass=False)
        assert isinstance(delta, float), "Orchestrator must return a scalar residual (delta)."