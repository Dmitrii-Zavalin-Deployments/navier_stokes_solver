# tests/operators/test_advection.py

import numpy as np

from src.step3.ops.advection import apply_advection_operator


def test_advection_operator_equivalence():
    nx, ny, nz = 5, 5, 5
    dx, dy, dz = 0.1, 0.1, 0.1
    v_n = np.random.rand(3, nx, ny, nz)
    f = np.random.rand(nx, ny, nz)
    
    # --- AUDIT: Loop-based reference ---
    ref = np.zeros((nx-2, ny-2, nz-2))
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            for k in range(1, nz-1):
                # Using cell-centered velocity
                u_i = (v_n[0, i+1, j, k] + v_n[0, i-1, j, k]) / 2
                v_i = (v_n[1, i, j+1, k] + v_n[1, i, j-1, k]) / 2
                w_i = (v_n[2, i, j, k+1] + v_n[2, i, j, k-1]) / 2
                
                dfdx = (f[i+1, j, k] - f[i-1, j, k]) / (2 * dx)
                dfdy = (f[i, j+1, k] - f[i, j-1, k]) / (2 * dy)
                dfdz = (f[i, j, k+1] - f[i, j, k-1]) / (2 * dz)
                
                ref[i-1, j-1, k-1] = u_i*dfdx + v_i*dfdy + w_i*dfdz
    
    # --- PRODUCTION ---
    opt = apply_advection_operator(v_n, f, dx, dy, dz)
    
    # --- PROOF ---
    assert np.allclose(ref, opt)