# tests/property_integrity/test_vertical_inheritance.py


# Core Logic
from src.step1.orchestrate_step1 import orchestrate_step1
from src.common.simulation_context import SimulationContext
from src.common.solver_config import SolverConfig

# Factory Functions (The "Recipes")
from tests.helpers.solver_input_schema_dummy import create_validated_input
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy

# Mock Config Data (Rule 5: Explicit numerical settings)
MOCK_CONFIG = {
    "ppe_tolerance": 1e-6,
    "ppe_atol": 1e-10,
    "ppe_max_iter": 1000,
    "ppe_omega": 1.5
}

class TestVerticalIntegrity:
    """
    Vertical Integrity Mandate (Rule 5):
    Verifies the 1:1 transformation from Input to Step 1 State.
    """

    def test_input_to_step1_pipeline(self):
        # 1. Scale
        NX, NY, NZ = 4, 4, 4
        
        # 2. Setup Dummies
        input_dummy = create_validated_input(nx=NX, ny=NY, nz=NZ)
        expected_state = make_step1_output_dummy(nx=NX, ny=NY, nz=NZ)
        
        # 3. Direct Context Assembly
        # We manually hydrate SolverConfig and wrap it with input_dummy
        config_obj = SolverConfig(**MOCK_CONFIG)
        context = SimulationContext(input_data=input_dummy, config=config_obj)
        
        # 4. Execute Orchestrator
        actual_state = orchestrate_step1(context)
        
        # 5. Parity Audit
        actual_dict = actual_state.to_dict()
        expected_dict = expected_state.to_dict()
        
        print(f"\n" + "="*60)
        print(f"VERTICAL INTEGRITY AUDIT: {NX}x{NY}x{NZ}")
        print("="*60)
        print(f"Actual Iteration: {actual_state.iteration}")
        
        # 6. Assertion
        assert actual_dict == expected_dict, "Step 1 Output Drift Detected!"
        
        print("SUCCESS: actual_state matches expected_state exactly.")
        print("="*60)