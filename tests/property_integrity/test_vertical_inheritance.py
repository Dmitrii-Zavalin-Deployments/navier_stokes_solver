# tests/property_integrity/test_vertical_inheritance.py


# Core Logic
from src.common.simulation_context import SimulationContext
from src.common.solver_config import SolverConfig
from src.step1.orchestrate_step1 import orchestrate_step1
from src.step2.orchestrate_step2 import orchestrate_step2
from src.step3.orchestrate_step3 import orchestrate_step3
from src.step4.orchestrate_step4 import orchestrate_step4

# Factory Functions (The "Recipes")
from tests.helpers.solver_input_schema_dummy import create_validated_input
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy

# Mock Config Data (Rule 5: Explicit numerical settings)
MOCK_CONFIG = {
    "ppe_tolerance": 1e-6,
    "ppe_atol": 1e-10,
    "ppe_max_iter": 1000,
    "ppe_omega": 1.5,
    "dt": 0.01
}

def assert_structural_parity(actual, expected, path=""):
    """
    Knowledge Gate Helper: Recursively verifies that keys and data types match.
    Ignores specific values to focus on Pipeline Integrity (Rule 5).
    """
    actual_keys = set(actual.keys())
    expected_keys = set(expected.keys())
    
    assert actual_keys == expected_keys, (
        f"Structure Break at '{path}': Missing or extra keys.\n"
        f"Diff: {actual_keys.symmetric_difference(expected_keys)}"
    )
    
    for key in actual:
        actual_val = actual[key]
        expected_val = expected[key]
        current_path = f"{path}.{key}" if path else key
        
        assert type(actual_val) is type(expected_val), (
            f"Type Mismatch at '{current_path}': "
            f"Expected {type(expected_val).__name__}, got {type(actual_val).__name__}"
        )
        
        if isinstance(actual_val, dict):
            assert_structural_parity(actual_val, expected_val, current_path)
        
        elif isinstance(actual_val, list):
            assert len(actual_val) == len(expected_val), (
                f"List Length Drift at '{current_path}': "
                f"Expected {len(expected_val)}, got {len(actual_val)}"
            )
            if len(actual_val) > 0:
                assert type(actual_val[0]) is type(expected_val[0]), (
                    f"List Element Type Mismatch at '{current_path}'"
                )

class TestVerticalIntegrity:
    """
    Vertical Integrity Mandate (Rule 5):
    Verifies data survival and structural alignment across the pipeline.
    """

    def test_input_to_step1_pipeline(self):
        """Phase 1: Validates Input -> Step 1 Orchestration"""
        NX, NY, NZ = 4, 4, 4
        
        input_dummy = create_validated_input(nx=NX, ny=NY, nz=NZ)
        expected_dummy = make_step1_output_dummy(nx=NX, ny=NY, nz=NZ)
        
        config_obj = SolverConfig(**MOCK_CONFIG)
        context = SimulationContext(input_data=input_dummy, config=config_obj)
        
        actual_state = orchestrate_step1(context)
        
        print(f"\n" + "-"*30)
        print(f"AUDIT: Input -> Step 1 ({NX}x{NY}x{NZ})")
        assert_structural_parity(actual_state.to_dict(), expected_dummy.to_dict())
        print("✅ Step 1 Structural Parity Secured")

    def test_step1_to_step2_pipeline(self):
        """Phase 2: Validates Step 1 -> Step 2 Orchestration"""
        NX, NY, NZ = 4, 4, 4
        
        # We start with the established Dummy for Step 1
        step1_dummy = make_step1_output_dummy(nx=NX, ny=NY, nz=NZ)
        expected_dummy = make_step2_output_dummy(nx=NX, ny=NY, nz=NZ)
        
        # Step 2 consumes the state directly
        actual_state = orchestrate_step2(step1_dummy)
        
        print(f"\n" + "-"*30)
        print(f"AUDIT: Step 1 -> Step 2 ({NX}x{NY}x{NZ})")
        assert_structural_parity(actual_state.to_dict(), expected_dummy.to_dict())
        print("✅ Step 2 Structural Parity Secured")
    
    def test_step2_to_step3_pipeline(self):
        """
        Phase 3: Validates Step 2 -> Step 3 Orchestration.
        Verifies that a StencilBlock can be processed for the Predictor Pass.
        """
        NX, NY, NZ = 4, 4, 4
        
        # 1. Setup Context (Needed for Step 3's physics parameters)
        input_dummy = create_validated_input(nx=NX, ny=NY, nz=NZ)
        config_obj = SolverConfig(**MOCK_CONFIG)
        context = SimulationContext(input_data=input_dummy, config=config_obj)
        
        # 2. Setup Dummies
        step2_state_dummy = make_step2_output_dummy(nx=NX, ny=NY, nz=NZ)
        expected_block_dummy = make_step3_output_dummy() # Usually represents one block
        
        # 3. Target a single block from the Step 2 Stencil Matrix
        # Step 3 operates on blocks, so we test the first one in the list
        sample_block = step2_state_dummy.stencil_matrix[0]
        
        # 4. Execute Step 3 Orchestrator (Predictor Pass / is_first_pass=True)
        # Based on main_solver.py: orchestrate_step3(block, context, is_first_pass)
        actual_block, delta = orchestrate_step3(
            block=sample_block,
            context=context,
            elasticity=step2_state_dummy.elasticity,
            is_first_pass=True
        )
        
        print(f"\n" + "-"*30)
        print(f"AUDIT: Step 2 -> Step 3 Block Integrity")
        print(f"Initial Convergence Delta: {delta}")
        
        # 5. Structural Assertion
        # Verify the block returned from Step 3 matches the Step 3 Dummy structure
        assert_structural_parity(actual_block.to_dict(), expected_block_dummy.to_dict())
        
        print("✅ Step 3 Structural Parity Secured (Predictor Pass)")

    def test_step3_to_step4_pipeline(self):
        """
        Phase 4: Validates Step 3 -> Step 4 Orchestration (Boundary Enforcement).
        Ensures the micro-level StencilBlock maintains schema parity after 
        applying physical boundary constraints (u, v, w, p).
        """
        NX, NY, NZ = 4, 4, 4
        
        # 1. Setup Context (Rule 4: SSoT)
        input_dummy = create_validated_input(nx=NX, ny=NY, nz=NZ)
        config_obj = SolverConfig(**MOCK_CONFIG)
        context = SimulationContext(input_data=input_dummy, config=config_obj)
        
        # 2. Assemble State up to Step 2
        state = orchestrate_step2(orchestrate_step1(context))
        
        # 3. Target: Select the first StencilBlock
        sample_block = state.stencil_matrix[0]
        
        # 4. EXECUTE: Step 4 Orchestrator
        actual_block = orchestrate_step4(
            block=sample_block,
            context=context,
            state_grid=state.grid,
            state_bc_manager=state.boundary_conditions
        )
        
        # 5. AUDIT: Compare against the Step 4 Micro-Dummy
        print(f"\n" + "-"*30)
        print(f"AUDIT: Step 3 -> Step 4 (Boundary Enforcement Parity)")
        
        expected_block_dummy = make_step4_output_dummy(nx=NX, ny=NY, nz=NZ, block_index=0)
        
        # Convert actual results to dict (standard serialization)
        actual_dict = actual_block.to_dict()
        
        # Prepare the expected dict and force recursion for all neighbor cells
        # This prevents "SimpleCellMock vs dict" type mismatches
        expected_dict = expected_block_dummy.to_dict()
        cell_keys = ['center', 'i_minus', 'i_plus', 'j_minus', 'j_plus', 'k_minus', 'k_plus']
        
        for key in cell_keys:
            if key in expected_dict:
                cell_obj = getattr(expected_block_dummy, key)
                if hasattr(cell_obj, 'to_dict'):
                    expected_dict[key] = cell_obj.to_dict()

        # Final structural validation
        assert_structural_parity(actual_dict, expected_dict)
        
        print("✅ Step 4 Structural Parity Secured")
    
    def test_step4_to_step5_pipeline(self):
        """
        Phase 5: Validates Step 4 -> Step 5 Orchestration (State Integration).
        Ensures that after Math (Step 3) and Boundaries (Step 4), 
        Step 5 can correctly consolidate the StencilMatrix back into Global Fields.
        """
        from src.step5.orchestrate_step5 import orchestrate_step5
        from tests.helpers.solver_step5_output_dummy import make_step5_output_dummy

        NX, NY, NZ = 4, 4, 4
        
        # 1. Setup Context & Base State
        input_dummy = create_validated_input(nx=NX, ny=NY, nz=NZ)
        config_obj = SolverConfig(**MOCK_CONFIG)
        context = SimulationContext(input_data=input_dummy, config=config_obj)
        
        # Assemble via Step 1 & 2
        state = orchestrate_step2(orchestrate_step1(context))
        
        # 2. SIMULATE ONE TIME-LOOP ITERATION (Rule 9: In-place mutation)
        # We process the entire matrix through Step 3 AND Step 4
        for block in state.stencil_matrix:
            block, _ = orchestrate_step3(block, context=context, elasticity=state.elasticity, is_first_pass=True)
            block, _ = orchestrate_step3(block, context=context, is_first_pass=True)
            
            # B. Boundary Enforcement (Step 4)
            orchestrate_step4(
                block=block, 
                context=context, 
                state_grid=state.grid, 
                state_bc_manager=state.boundary_conditions
            )
        
        # 3. EXECUTE: Step 5 (Global Integration)
        # Step 5 takes the mutated state and updates the global field buffers
        actual_final_state = orchestrate_step5(state, context)
        
        # 4. AUDIT
        print(f"\n" + "-"*30)
        print(f"AUDIT: Step 4 -> Step 5 (Global Integration Integrity)")
        
        expected_state_dummy = make_step5_output_dummy(nx=NX, ny=NY, nz=NZ)
        
        assert_structural_parity(
            actual_final_state.to_dict(), 
            expected_state_dummy.to_dict()
        )
        
        print("✅ Step 5 Structural Parity Secured")