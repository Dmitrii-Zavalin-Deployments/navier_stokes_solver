# src/main_solver.py

import json
import logging
import sys
from pathlib import Path

import jsonschema
import numpy as np

from src.common.archive_service import archive_simulation_artifacts
from src.common.elasticity import ElasticManager
from src.common.simulation_context import SimulationContext
from src.step1.orchestrate_step1 import orchestrate_step1
from src.step2.orchestrate_step2 import orchestrate_step2
from src.step3.orchestrate_step3 import orchestrate_step3
from src.step4.orchestrate_step4 import orchestrate_step4
from src.step5.orchestrate_step5 import orchestrate_step5

np.seterr(all="raise")

DEBUG = False
logger = logging.getLogger("Solver.Main")
BASE_DIR = Path(__file__).resolve().parent.parent

def _load_simulation_context(input_path: str) -> SimulationContext:
    """Assembles physical input and numerical config into a unified context."""
    full_input_path = BASE_DIR / input_path
    config_path = BASE_DIR / "config.json"

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
    """Main Orchestrator with Unified Elastic Stability."""
    context = _load_simulation_context(input_path)

    # 1. VALIDATE INPUT
    SCHEMA_PATH = BASE_DIR / "schema/solver_input_schema.json"
    try:
        with open(SCHEMA_PATH) as f:
            schema = json.load(f)
        jsonschema.validate(instance=context.input_data.to_dict(), schema=schema)
        if DEBUG:
            print("DEBUG [Main]: Input schema validation passed.")
    except jsonschema.exceptions.ValidationError as e:
        print(f"!!! CONTRACT VIOLATION: {e.message}")
        raise

    # 2. ASSEMBLY
    state = orchestrate_step1(context)
    state = orchestrate_step2(state)

    # 3. STATE CONTRACT VALIDATION
    try:
        state.validate_against_schema(str(SCHEMA_PATH))
        if DEBUG:
            print("DEBUG [Main]: State validation passed.")
    except jsonschema.exceptions.ValidationError as e:
        path_str = ".".join([str(p) for p in e.path])
        print(f"!!! CONTRACT VIOLATION at {path_str}: {e.message}")
        raise

    # 4. ELASTICITY ENGINE
    elasticity = ElasticManager(
        context.config,
        context.input_data.simulation_parameters.time_step,
    )

    # 5. MAIN EXECUTION LOOP
    while state.ready_for_time_loop:
        try:
            # A. PREDICTOR PASS
            for block in state.stencil_matrix:
                orchestrate_step3(block, context, elasticity, is_first_pass=True)
                orchestrate_step4(block, context, state.grid, state.boundary_conditions)

            # B. PPE ITERATION
            for _ in range(context.config.ppe_max_iter):
                max_delta = 0.0
                for block in state.stencil_matrix:
                    _, delta = orchestrate_step3(block, context, elasticity, is_first_pass=False)
                    orchestrate_step4(block, context, state.grid, state.boundary_conditions)
                    max_delta = max(max_delta, delta)

                if max_delta < context.config.ppe_tolerance:
                    break
            
            # C. ADVANCE
            state.iteration += 1
            state.time += elasticity.dt
            state = orchestrate_step5(state, context)

            if DEBUG and state.iteration % 10 == 0:
                print(f"DEBUG [Main]: Step {state.iteration} | Time {state.time:.4f} | dt {elasticity.dt:.2e}")

            if state.time >= context.input_data.simulation_parameters.total_time:
                state.ready_for_time_loop = False
            
            # SUCCESS PATH
            elasticity.stabilization(is_needed=False)

        except ArithmeticError:
            # FAILURE PATH: Update dt and retry the SAME time step
            elasticity.stabilization(is_needed=True)

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