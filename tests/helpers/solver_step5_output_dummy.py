# tests/helpers/solver_step5_output_dummy.py

from src.common.field_schema import FI
from src.common.simulation_context import SimulationContext
from src.common.solver_config import SolverConfig
from src.step1.orchestrate_step1 import orchestrate_step1
from src.step2.orchestrate_step2 import orchestrate_step2
from tests.helpers.solver_input_schema_dummy import create_validated_input


def make_step5_output_dummy(nx: int = 4, ny: int = 4, nz: int = 4):
    """
    Returns a 'frozen' prototype representing the global system state 
    after Step 3 (Math) and Step 4 (Boundaries) have mutated the memory.
    """
    # 1. Reconstruct global foundation
    MOCK_CONFIG = {
        "ppe_tolerance": 1e-6,
        "ppe_atol": 1e-10,
        "ppe_max_iter": 1000,
        "ppe_omega": 1.5
    }
    
    input_dummy = create_validated_input(nx=nx, ny=ny, nz=nz)
    config_obj = SolverConfig(**MOCK_CONFIG)
    context = SimulationContext(input_data=input_dummy, config=config_obj)
    
    # Assembly via Step 1 & 2 (Allocates the Hybrid Memory Foundation)
    state = orchestrate_step2(orchestrate_step1(context))
    
    # 2. Populate Foundation (Rule 9: Hybrid Memory Foundation)
    # This mimics the outcome of Step 3 + Step 4 loop
    data = state.fields.data
    
    # Set Primary Velocities (Result of Step 4 boundary enforcement)
    data[:, FI.VX] = 0.5
    data[:, FI.VY] = 0.5
    data[:, FI.VZ] = 0.5
    
    # Set Intermediate Velocities (Predictor results from Step 3)
    data[:, FI.VX_STAR] = 0.51
    data[:, FI.VY_STAR] = 0.51
    data[:, FI.VZ_STAR] = 0.51
    
    # Set Pressure Fields (Post-PPE synchronization)
    data[:, FI.P] = 0.01
    data[:, FI.P_NEXT] = 0.01

    # 3. Set Iteration/Time context
    state.iteration = 1
    state.time = input_dummy.simulation_parameters.time_step 
    
    # 4. Manifest Integrity (Archival State)
    if not hasattr(state, 'manifest'):
        class ManifestDummy: pass
        state.manifest = ManifestDummy()
    
    state.manifest.saved_snapshots = ["output/snapshot_0000.h5"]
    state.manifest.output_directory = "output/"
    state.ready_for_time_loop = True
    
    return state