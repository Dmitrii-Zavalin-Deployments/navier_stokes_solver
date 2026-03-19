# src/main_solver.py

import json
import sys
from pathlib import Path

import jsonschema

from src.common.archive_service import archive_simulation_artifacts
from src.common.elasticity import ElasticManager  # Moved to common
from src.common.simulation_context import SimulationContext
from src.step1.orchestrate_step1 import orchestrate_step1
from src.step2.orchestrate_step2 import orchestrate_step2
from src.step3.orchestrate_step3 import orchestrate_step3
from src.step4.orchestrate_step4 import orchestrate_step4
from src.step5.orchestrate_step5 import orchestrate_step5

# Global Debug Toggle: Rule 7 requires high-res logging for math
DEBUG = True
BASE_DIR = Path(__file__).resolve().parent.parent

def _load_simulation_context(input_path: str) -> SimulationContext:
    """Assembles physical input and numerical config into a unified context."""
    full_input_path = BASE_DIR / input_path
    config_path = BASE_DIR / "config.json"
    
    # Rule 5: Explicit or Error. No fallbacks/defaults.
    if not full_input_path.exists():
        raise FileNotFoundError(f"Input file missing at {full_input_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"config.json required at {config_path}")

    with open(full_input_path) as f:
        input_data = json.load(f)
    with open(config_path) as f:
        config_data = json.load(f)
        
    return SimulationContext.create(input_data, config_data)

def run_solver(input_path: str) -> str:
    """Main Orchestrator with Elastic Stability."""
    
    context = _load_simulation_context(input_path)

    # 1. PRE-EXECUTION FIREWALL: Validate Input Schema
    SCHEMA_PATH = BASE_DIR / "schema/solver_input_schema.json"
    try:
        with open(SCHEMA_PATH) as f:
            schema = json.load(f)
        jsonschema.validate(instance=context.input_data.to_dict(), schema=schema)
        if DEBUG:
            print(f"DEBUG [Main]: ✅ Input schema validation passed.")
    except jsonschema.exceptions.ValidationError as e:
        print(f"!!! CONTRACT VIOLATION: {e.message}")
        raise

    # 2. ASSEMBLY via Orchestrators (Foundation logic)
    state = orchestrate_step1(context)
    state = orchestrate_step2(state)

    # 3. FIREWALL: State Contract Validation (Post-Assembly, Rule 4/SSoT)
    try:
        state.validate_against_schema(str(SCHEMA_PATH))
        if DEBUG:
            print("DEBUG [Main]: ✅ State validation passed.")
    except jsonschema.exceptions.ValidationError as e:
        path_str = '.'.join([str(p) for p in e.path])
        print(f"!!! CONTRACT VIOLATION at {path_str}: {e.message}")
        raise
    
    # 4. ELASTICITY ENGINE (Numerical SSoT)
    # We pass context.config directly as elasticity manages numerical behavior
    elasticity = ElasticManager(context.config)

    # 5. MAIN EXECUTION LOOP
    while state.ready_for_time_loop:
        try:
            # A. PREDICTOR PASS
            # Rule 4: block.dt is internally synced with elasticity.dt
            for block in state.stencil_matrix:
                orchestrate_step3(block, context, elasticity, is_first_pass=True)
                orchestrate_step4(block, context, state.grid, state.boundary_conditions)
            
            # B. ITERATIVE SOLVER (PPE)
            for _ in range(elasticity.max_iter):
                max_delta = 0.0
                for block in state.stencil_matrix:
                    _, delta = orchestrate_step3(block, context, elasticity, is_first_pass=False)
                    orchestrate_step4(block, context, state.grid, state.boundary_conditions)
                    max_delta = max(max_delta, delta)
                
                # Performance optimization: Exit PPE loop if tolerance met
                if max_delta < context.config.ppe_tolerance:
                    break
            
            # C. VALIDATE & COMMIT (Transactional Gate)
            # Rule 4/9: Merges trial buffers only if the time-step is numerically valid
            if not elasticity.validate_and_commit(state):
                raise ArithmeticError("Numerical instability detected in trial buffers.")
        
            # D. ADVANCE (Physical & Temporal)
            state.iteration += 1
            state.time += elasticity.dt 
            state = orchestrate_step5(state, context)
            
            # Heal parameters if simulation is running smoothly
            elasticity.gradual_recovery()

            if DEBUG and state.iteration % 10 == 0:
                print(f"DEBUG [Main]: Step {state.iteration} | Time {state.time:.4f} | dt {elasticity.dt:.2e}")

        except ArithmeticError as e:
            if DEBUG:
                print(f"DEBUG [Main]: ⚠️ {str(e)} -> Triggering Panic Mode.")
            elasticity.apply_panic_mode()
            continue # Retry the same time-step with safer parameters
        
        # Termination check
        if state.time >= context.input_data.simulation_parameters.total_time:
            state.ready_for_time_loop = False

    # 6. ARCHIVING TRIGGER (Rule 4: Atomic lifecycle completion)
    return archive_simulation_artifacts(state)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/main_solver.py <input_json_path>")
        sys.exit(1)
    
    try:
        zip_path = run_solver(sys.argv[1])
        print(f"Pipeline complete. Artifacts archived at: {zip_path}")
        sys.exit(0)
    except Exception as e:
        print(f"FATAL PIPELINE ERROR: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)