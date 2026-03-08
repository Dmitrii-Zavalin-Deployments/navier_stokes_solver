# tests/solver/test_projection_method.py

import numpy as np

from src.step3.ops.divergence import divergence_v_star
from src.step3.solver.corrector import compute_corrector_step

# We will assume we have these components ready
from src.step3.solver.predictor import compute_predictor_step


def test_incompressibility_constraint():
    # 1. Setup a known state
    nx, ny, nz = 10, 10, 10
    dx, dy, dz = 0.1, 0.1, 0.1
    # Random velocity
    v_n = np.random.rand(3, nx, ny, nz)
    
    # 2. Run the pipeline
    # Predictor -> Pressure Poisson -> Corrector
    v_star = compute_predictor_step(v_n, ...)
    p_next = solve_pressure_poisson(v_star, ...)
    v_next = compute_corrector_step(v_star, p_next, ...)
    
    # 3. AUDIT: The Divergence-Free Proof
    final_divergence = divergence_v_star(v_next, dx, dy, dz)
    
    # 4. ASSERT: The result must be near zero
    # Use a small tolerance due to floating point errors
    assert np.allclose(final_divergence, 0.0, atol=1e-10)