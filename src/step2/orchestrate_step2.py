# src/step2/orchestrate_step2.py

import json
from src.solver_state import SolverState
from .operators import build_numerical_operators
from .advection import build_advection_stencils

def orchestrate_step2(state: SolverState) -> SolverState:
    """
    Step 2 Orchestrator: Mathematical Readiness.
    
    Rule 5: Explicit or Error. No defaults/fallbacks. 
    Ensures SSoT for solver configuration and operator hydration.
    """
    
    # --- Point 2: Calculate & Prepare ---
    try:
        with open("config.json", "r") as f:
            external_config = json.load(f)
        
        settings = external_config["solver_settings"]
        
        # Explicit mapping: No .get() allowed per Phase C mandate.
        # These must exist in config.json or the solver will (rightfully) crash.
        state.config.ppe_atol = settings["ppe_atol"]
        state.config.ppe_max_iter = settings["ppe_max_iter"]
        
    except FileNotFoundError:
        raise FileNotFoundError(
            "Critical Error: 'config.json' not found. "
            "A valid configuration is required to initialize solver tolerances."
        )
    except KeyError as e:
        raise KeyError(
            f"Critical Error: Missing required solver setting {e} in 'config.json'. "
            "Verify ppe_atol and ppe_max_iter are defined."
        )
    except json.JSONDecodeError:
        raise ValueError("Critical Error: 'config.json' contains invalid JSON syntax.")

    # Delegate math to worker files
    build_numerical_operators(state)
    build_advection_stencils(state)

    # --- Point 3: Insertion & State Baseline (THE HANDSHAKE) ---
    # CONSTITUTIONAL ALIGNMENT: 
    # Assign to '_A' to match the attribute expected by Step 3's solve_pressure.
    state.ppe._A = state.operators.laplacian
    state.ppe.preconditioner = None 

    # Initialize Health Vitals (Baseline Reset)
    state.health.max_u = 0.0
    state.health.divergence_norm = 0.0
    state.health.is_stable = True
    state.health.post_correction_divergence_norm = 0.0

    state.ready_for_time_loop = True 
    
    return state