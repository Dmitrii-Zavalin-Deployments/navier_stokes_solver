# tests/property_integrity/test_vertical_inheritance.py

import pytest
import json

# Core Logic
from src.step1.orchestrate_step1 import orchestrate_step1
from src.common.simulation_context import SimulationContext

# Factory Functions (The "Recipes")
from tests.helpers.solver_input_schema_dummy import create_validated_input
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy


class TestVerticalIntegrity:
    """
    Vertical Integrity Mandate (Rule 5):
    Uses factory-aligned dummies to verify the 1:1 transformation.
    Ensures that SolverInput is correctly ingested into SolverState.
    """

    def test_input_to_step1_pipeline(self):
        # 1. Define the simulation scale for this test
        NX, NY, NZ = 4, 4, 4
        
        # 2. Instantiate the Dummies inside the test (Signal-Aligned)
        input_dummy = create_validated_input(nx=NX, ny=NY, nz=NZ)
        expected_state = make_step1_output_dummy(nx=NX, ny=NY, nz=NZ)
        
        # 3. Prepare the Context Wrapper (Phase C Requirement)
        # The orchestrator expects a context object containing the input_data
        context = SimulationContext()
        context.input_data = input_dummy
        
        # 4. Run the actual orchestrator
        actual_state = orchestrate_step1(context)
        
        # 5. Observation Phase
        # We use json.dumps to make the dict output readable in the CI logs
        actual_dict = actual_state.to_dict()
        expected_dict = expected_state.to_dict()
        
        print(f"\n" + "="*60)
        print(f"VERTICAL INTEGRITY AUDIT: {NX}x{NY}x{NZ}")
        print("="*60)
        print(f"Actual Iteration: {actual_state.iteration}")
        print(f"Ghost Grid Shape: {actual_state.grid.nx + 2}x{actual_state.grid.ny + 2}x{actual_state.grid.nz + 2}")
        
        # 6. The "Full-Tree" Parity Check
        # Rule 5: Compare the entire object hierarchy via dictionary serialization
        assert actual_dict == expected_dict, (
            "Step 1 Output Drift Detected! The produced SolverState does not match the dummy. "
            "Check captured stdout for dictionary diff."
        )
        
        print("SUCCESS: actual_state matches expected_state exactly.")
        print("="*60)