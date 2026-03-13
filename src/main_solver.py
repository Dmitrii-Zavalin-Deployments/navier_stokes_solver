# src/main_solver.py

import json
import sys
from pathlib import Path

import jsonschema

from src.common.archive_service import archive_simulation_artifacts
from src.common.simulation_context import SimulationContext
from src.step1.orchestrate_step1 import orchestrate_step1
from src.step2.orchestrate_step2 import orchestrate_step2
from src.step3.orchestrate_step3 import orchestrate_step3
from src.step4.orchestrate_step4 import orchestrate_step4
from src.step5.orchestrate_step5 import orchestrate_step5

# Global Debug Toggle
DEBUG = True
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

def run_solver(input_path: str):
    """Orchestrates the physics pipeline using the unified SimulationContext."""
    
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

    # 2. ASSEMBLY via Orchestrators
    state = orchestrate_step1(context)
    state = orchestrate_step2(state)

    # 3. MAIN EXECUTION LOOP
    state.ready_for_time_loop = True
    
    while state.ready_for_time_loop:
        # A. PREDICTOR PASS
        for block in state.stencil_matrix:
            block, _ = orchestrate_step3(block, context=context, is_first_pass=True)
            # UPDATED: Passing context, grid, and boundary manager
            block = orchestrate_step4(block, context, state.grid, state.boundary_conditions)
        
        # B. ITERATIVE SOLVER & BOUNDARY PASS
        for _ in range(context.config.ppe_max_iter):
            max_delta = 0.0
            for block in state.stencil_matrix:
                block, delta = orchestrate_step3(block, context=context, is_first_pass=False)
                # UPDATED: Passing context, grid, and boundary manager
                block = orchestrate_step4(block, context, state.grid, state.boundary_conditions)
                max_delta = max(max_delta, delta)
            
            if max_delta < context.config.ppe_tolerance:
                break
        
        state.iteration += 1
        state.time += state.sim_params.time_step
        state = orchestrate_step5(state, context)
        
        if state.time >= state.sim_params.total_time:
            state.ready_for_time_loop = False

    return state

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Input file path required.")
        sys.exit(1)
    
    try:
        final_state = run_solver(sys.argv[1])
        zip_path = archive_simulation_artifacts(final_state)
        print(zip_path)
        sys.exit(0)
    except Exception as e:
        print(f"FATAL PIPELINE ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)