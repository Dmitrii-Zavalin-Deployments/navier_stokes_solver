# tests/operators/test_laplacian.py
import numpy as np

from src.step3.ops.laplacian import laplacian_p_n_plus_1, laplacian_v_n


def test_laplacian_equivalence():
    nx, ny, nz = 5, 5, 5
    dx, dy, dz = 0.1, 0.1, 0.1
    
    # 1. Setup Test Data
    v_n = np.random.rand(3, nx, ny, nz)
    p = np.random.rand(nx, ny, nz)
    
    # --- AUDIT: Loop-based reference ---
    def audit_lap(f):
        ref = np.zeros((nx-2, ny-2, nz-2))
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                for k in range(1, nz-1):
                    ref[i-1, j-1, k-1] = (
                        (f[i+1, j, k] - 2*f[i, j, k] + f[i-1, j, k]) / dx**2 +
                        (f[i, j+1, k] - 2*f[i, j, k] + f[i, j-1, k]) / dy**2 +
                        (f[i, j, k+1] - 2*f[i, j, k] + f[i, j, k-1]) / dz**2
                    )
        return ref

    # --- EXECUTION: Optimized production code ---
    opt_v = laplacian_v_n(v_n, dx, dy, dz)
    opt_p = laplacian_p_n_plus_1(p, dx, dy, dz)
    
    # --- PROOF: Assert equivalence ---
    assert np.allclose(audit_lap(v_n[0]), opt_v[0]) # Test component u
    assert np.allclose(audit_lap(p), opt_p)          # Test scalar p