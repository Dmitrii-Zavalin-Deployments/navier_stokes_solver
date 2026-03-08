# tests/operators/test_divergence.py

import numpy as np

from src.step3.ops.divergence import divergence_v_star


def test_divergence_v_star_equivalence():
    nx, ny, nz = 5, 5, 5
    dx, dy, dz = 0.1, 0.1, 0.1
    # v_star has 3 components: [u, v, w]
    v_star = np.random.rand(3, nx, ny, nz)
    u, v, w = v_star[0], v_star[1], v_star[2]
    
    # --- AUDIT: Loop-based reference ---
    ref = np.zeros((nx-2, ny-2, nz-2))
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            for k in range(1, nz-1):
                ref[i-1, j-1, k-1] = (
                    (u[i+1, j, k] - u[i-1, j, k]) / (2 * dx) +
                    (v[i, j+1, k] - v[i, j-1, k]) / (2 * dy) +
                    (w[i, j, k+1] - w[i, j, k-1]) / (2 * dz)
                )
    
    # --- PRODUCTION: Optimized slicing ---
    opt = divergence_v_star(v_star, dx, dy, dz)
    
    # --- PROOF ---
    assert np.allclose(ref, opt)