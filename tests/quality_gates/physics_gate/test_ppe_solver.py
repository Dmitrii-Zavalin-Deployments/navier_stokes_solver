# tests/quality_gates/physics_gate/test_ppe_solver.py

import pytest

from src.common.field_schema import FI
from src.step3.ppe_solver import solve_pressure_poisson_step
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


def setup_ppe_block(dt=1.0, rho=1.0, dx=1.0):
    """
    Sets up a StencilBlock for PPE testing with Unit Geometry.
    """
    # Use our standard dummy (which uses the real buffer/stencil logic)
    state = make_step2_output_dummy(nx=4, ny=4, nz=4)
    block = state.stencil_matrix[10] # Grab a center-ish block

    # Bypass read-only properties to force Unit Geometry
    object.__setattr__(block, '_dt', float(dt))
    object.__setattr__(block, '_rho', float(rho))
    object.__setattr__(block, '_dx', float(dx))
    object.__setattr__(block, '_dy', float(dx))
    object.__setattr__(block, '_dz', float(dx))

    # Initialize all fields in the buffer to 0.0
    block.center.fields_buffer.fill(0.0)
    
    return block

def test_ppe_unit_convergence():
    """
    Verifies that p_new is calculated correctly when divergence is the only source.
    Formula simplified (with dx=1, P=0, omega=1):
    p_new = (1/stencil_denom) * (sum_neighbors - (rho/dt)*div_v)
    """
    # 1. Setup (dx=1.0, dt=1.0, rho=1.0)
    block = setup_ppe_block(dt=1.0, rho=1.0, dx=1.0)
    omega = 1.0 # Pure Gauss-Seidel, no relaxation overhead
    
    # 2. Set linear velocity field to create known divergence
    # div_v = (vx_ip - vx_im)/2 + ... 
    # Let's make div_v = 1.0
    block.i_plus.set_field(FI.VX_STAR, 2.0)
    block.i_minus.set_field(FI.VX_STAR, 0.0)
    # grad_x = (2-0)/2 = 1.0. Others are 0.
    
    # 3. Calculations
    # stencil_denom = 2 * (1/1 + 1/1 + 1/1) = 6.0
    # rhs = (rho/dt) * (div_v - 0) = 1.0 * (1.0 - 0) = 1.0
    # sum_neighbors = 0 (all P_NEXT are 0)
    # p_new = (1/6) * (0 - 1.0) = -0.16666666666666666
    
    residual = solve_pressure_poisson_step(block, omega)
    
    expected_p = -1.0 / 6.0
    actual_p = block.center.get_field(FI.P_NEXT)
    
    assert actual_p == pytest.approx(expected_p, rel=1e-12)
    assert residual == pytest.approx(abs(expected_p), rel=1e-12)

def test_ppe_rhie_chow_teamwork():
    """
    Checks if Rhie-Chow stabilization properly offsets the RHS.
    If div_v == rhie_chow_term, RHS should be 0.
    """
    block = setup_ppe_block(dt=1.0, rho=1.0, dx=1.0)
    
    # Set div_v = 1.0
    block.i_plus.set_field(FI.VX_STAR, 2.0)
    block.i_minus.set_field(FI.VX_STAR, 0.0)
    
    # Set P fields such that laplacian(P) = 1.0 (to match div_v)
    # lap = (P_ip - 2P_c + P_im) / dx^2
    # (1 - 0 + 0) / 1 = 1.0
    block.i_plus.set_field(FI.P, 1.0)
    block.center.set_field(FI.P, 0.0)
    block.i_minus.set_field(FI.P, 0.0)
    
    # rhs = (rho/dt) * (div_v_star - (dt * lap_p))
    # rhs = 1.0 * (1.0 - (1.0 * 1.0)) = 0.0
    
    solve_pressure_poisson_step(block, omega=1.0)
    
    # If RHS is 0 and neighbors are 0, P_next must stay 0
    assert block.center.get_field(FI.P_NEXT) == 0.0

def test_ppe_sor_relaxation():
    """
    Verifies that omega (relaxation factor) scales the update correctly.
    """
    block = setup_ppe_block(dt=1.0, rho=1.0, dx=1.0)
    omega = 0.5 
    
    # Set P_old = 10.0
    block.center.set_field(FI.P_NEXT, 10.0)
    
    # Force RHS = 0, sum_neighbors = 0
    # p_new = (1 - 0.5)*10.0 + (0.5 / 6.0) * (0 - 0) = 5.0
    
    solve_pressure_poisson_step(block, omega)
    
    assert block.center.get_field(FI.P_NEXT) == pytest.approx(5.0)