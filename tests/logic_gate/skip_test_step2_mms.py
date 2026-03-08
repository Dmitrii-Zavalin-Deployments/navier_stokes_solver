# tests/logic_gate/test_step2_mms.py

import numpy as np

from src.step2.orchestrate_step2 import orchestrate_step2
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy


def test_logic_gate_2_quadratic_laplacian():
    """
    LOGIC GATE 2: Quadratic Field MMS.
    Analytical Truth: ∇²p = 6.0
    """
    nx, ny, nz = 4, 4, 4
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # CONSTITUTIONAL ALIGNMENT: Use is_fluid mask
    state.masks.is_fluid = np.ones((nx, ny, nz), dtype=int)
    
    state = orchestrate_step2(state)
    
    L = state.operators.laplacian
    dx, dy, dz = state.grid.dx, state.grid.dy, state.grid.dz
    
    p_analytic = np.zeros((nx, ny, nz), order='F')
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                x, y, z = (i+0.5)*dx, (j+0.5)*dy, (k+0.5)*dz
                p_analytic[i,j,k] = x**2 + y**2 + z**2
    
    res_vec = L.dot(p_analytic.flatten(order='F'))
    res_3d = res_vec.reshape((nx, ny, nz), order='F')
    
    # Success Metric: 6.0 on internal nodes
    internal_res = res_3d[1:-1, 1:-1, 1:-1]
    np.testing.assert_allclose(internal_res, 6.0, rtol=1e-10)