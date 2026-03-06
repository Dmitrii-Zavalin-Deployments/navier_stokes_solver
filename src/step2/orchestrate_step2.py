# src/step2/orchestrate_step2.py

import json

from src.solver_state import SolverState

from .advection import build_advection_stencils
from .operators import build_numerical_operators


def orchestrate_step2(state: SolverState) -> SolverState:
    """
    Step 2 Orchestrator: Mathematical Readiness.
    Loads external configuration, builds discrete operators, and initializes health vitals.
    """
    
    # --- 1. CONFIG HYDRATION ---
    try:
        with open("config.json") as f:
            external_config = json.load(f)
        
        settings = external_config["solver_settings"]
        
        # MANDATORY TRIAD: Initializing solver parameters from external config
        state.config.ppe_atol = settings["ppe_atol"]
        state.config.ppe_tolerance = settings["ppe_tolerance"]
        state.config.ppe_max_iter = settings["ppe_max_iter"]
        
    except FileNotFoundError as e:
        raise FileNotFoundError("Critical Error: 'config.json' not found.") from e
    except KeyError as e:
        # Improved logging to show specifically which key is missing
        raise KeyError(f"Critical Error: Missing required solver setting {e} in 'config.json'.") from e
    except json.JSONDecodeError as e:
        # Improved logging to capture the JSON syntax error details
        raise ValueError(f"Critical Error: 'config.json' is not a valid JSON file. Details: {e}") from e

    # --- 2. OPERATOR GENERATION ---
    # Delegate math to specialized worker modules
    build_numerical_operators(state)
    build_advection_stencils(state)

    # --- 3. STATE HANDSHAKE & BASELINE ---
    # Scientific Handshake: Link the Laplacian to the PPE Solver
    state.ppe._A = state.operators.laplacian
    
    # Corrected attribute name (added underscore) to match SolverState and Scientific Tests
    state.ppe._preconditioner = None 

    # Initialize Health Vitals to baseline before the time loop starts
    state.health.max_u = 0.0
    state.health.divergence_norm = 0.0
    state.health.is_stable = True
    state.health.post_correction_divergence_norm = 0.0

    # Final readiness toggle for the Main Solver
    state.ready_for_time_loop = True 
    
    return state