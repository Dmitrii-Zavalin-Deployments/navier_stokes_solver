# tests/property_integrity/test_vertical_inheritance.py

import pytest
import numpy as np

# Core Logic
from src.step1.orchestrate_step1 import orchestrate_step1

# Factory Functions (The "Recipes")
from tests.helpers.solver_input_schema_dummy import create_validated_input
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy

class TestVerticalIntegrity:
    """
    Vertical Integrity Mandate (Rule 5):
    Uses factory-aligned dummies to verify the 1:1 transformation.
    """

    def test_input_to_step1_pipeline(self):
        # 1. Define the simulation scale for this test
        NX, NY, NZ = 4, 4, 4
        
        # 2. Instantiate the Dummies inside the test (Signal-Aligned)
        input_data = create_validated_input(nx=NX, ny=NY, nz=NZ)
        expected_state = make_step1_output_dummy(nx=NX, ny=NY, nz=NZ)
        
        # 3. Run the actual orchestrator
        actual_state = orchestrate_step1(input_data)
        
        # 4. Observation Phase (as you requested)
        print(f"\nAudit for Grid Size: {NX}x{NY}x{NZ}")
        print(f"Actual Iteration: {actual_state.iteration}")
        print(f"Expected Iteration: {expected_state.iteration}")
        
        # 5. The "Full-Tree" Parity Check
        # Using to_dict() for a quick high-level deep comparison
        assert actual_state.to_dict() == expected_state.to_dict()