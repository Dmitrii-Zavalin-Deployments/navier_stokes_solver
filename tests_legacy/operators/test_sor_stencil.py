# tests/operators/test_sor_stencil.py

import numpy as np

from src.step3.ops.sor_stencil import compute_sor_stencil


def test_sor_stencil_on_quadratic_field():
    # 1. Setup a small 3D grid
    nx, ny, nz = 5, 5, 5
    dx, dy, dz = 0.1, 0.1, 0.1
    dx2, dy2, dz2 = dx**2, dy**2, dz**2
    stencil_denom = 2.0 * (1/dx2 + 1/dy2 + 1/dz2)
    
    # 2. Create coordinates and p(x,y,z) = x^2 + y^2 + z^2
    x = np.linspace(0, (nx-1)*dx, nx)
    y = np.linspace(0, (ny-1)*dy, ny)
    z = np.linspace(0, (nz-1)*dz, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    p = X**2 + Y**2 + Z**2
    
    # 3. Set RHS to 0 (we are just testing the Laplacian part of the stencil)
    rhs = np.zeros((nx-2, ny-2, nz-2))
    
    # 4. Run the operator
    result = compute_sor_stencil(p, dx2, dy2, dz2, stencil_denom, rhs)
    
    # 5. Expected: Laplacian(x^2 + y^2 + z^2) = 6.0
    # Our operator computes: Laplacian(p) - RHS
    expected = 6.0 * np.ones_like(result)
    
    assert np.allclose(result, expected, atol=1e-10)

def test_sor_stencil_dimensions():
    # Ensure it correctly reduces the input field by 1 cell on each boundary
    p = np.random.rand(10, 10, 10)
    rhs = np.zeros((8, 8, 8))
    res = compute_sor_stencil(p, 1, 1, 1, 6, rhs)
    assert res.shape == (8, 8, 8)