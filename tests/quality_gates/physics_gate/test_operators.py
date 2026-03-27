# tests/quality_gates/physics_gate/test_operators.py

import pytest
import numpy as np
from src.common.field_schema import FI
from src.step3.ops.gradient import compute_local_gradient_p
from src.step3.ops.laplacian import compute_local_laplacian
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy

def test_gate_3a_central_difference_lock():
    """
    Gate 3.A: Central Difference Lock Audit
    Mathematical Check: ∂p/∂x ≈ (p_ip - p_im) / 2Δx
    Compliance: Rule 7 (Fail-Fast math audit)
    """
    # 1. Setup: Grab an interior block from the 4x4x4 core
    state = make_step2_output_dummy(nx=4, ny=4, nz=4)
    block = state.stencil_matrix[10] 
    dx = block.dx
    
    # 2. Impose a linear pressure gradient: p = 10 * x
    # p_im (left) = 0.0, p_ip (right) = 20.0
    # Center value is irrelevant for central difference gradient.
    block.i_minus.set_field(FI.P, 0.0)
    block.i_plus.set_field(FI.P, 20.0)
    
    # 3. Action: Compute gradient
    grad_p = compute_local_gradient_p(block, field_id=FI.P)
    grad_x = grad_p[0]
    
    # 4. Verification: Check the 2*dx divisor (The Lock)
    expected = (20.0 - 0.0) / (2.0 * dx)
    
    assert np.isclose(grad_x, expected), (
        f"Central Difference Lock Breach: ∂p/∂x is {grad_x}, "
        f"expected {expected} (Check for 1.0*dx divisor error)."
    )

def test_gate_3b_laplacian_second_order():
    """
    Gate 3.B: Second-Order Discretization Audit
    Mathematical Check: ∂²f/∂x² ≈ (f_ip - 2fc + f_im) / Δx²
    Compliance: Rule 8 (Centralized Logic)
    """
    # 1. Setup
    state = make_step2_output_dummy(nx=4, ny=4, nz=4)
    block = state.stencil_matrix[10]
    dx = block.dx
    
    # 2. Impose a quadratic field: f(x) = x^2 
    # At local points relative to center: x-dx, x, x+dx
    # Let center x=1.0, dx=0.25 -> im=0.75, ip=1.25
    f_c = 1.0**2
    f_im = (1.0 - dx)**2
    f_ip = (1.0 + dx)**2
    
    block.center.set_field(FI.P, f_c)
    block.i_minus.set_field(FI.P, f_im)
    block.i_plus.set_field(FI.P, f_ip)
    
    # Zero out other axes to isolate X-component
    block.j_minus.set_field(FI.P, f_c); block.j_plus.set_field(FI.P, f_c)
    block.k_minus.set_field(FI.P, f_c); block.k_plus.set_field(FI.P, f_c)
    
    # 3. Action: Compute Laplacian
    lap = compute_local_laplacian(block, FI.P)
    
    # 4. Verification: ∂²(x²)/∂x² = 2.0
    # Expected discrete: (f_ip - 2*f_c + f_im) / dx^2
    expected_x = (f_ip - 2.0 * f_c + f_im) / (dx**2)
    # Since other axes are constant, their derivatives are 0.
    
    assert np.isclose(lap, expected_x), (
        f"Laplacian Fidelity Breach: {lap} != {expected_x}. "
        "Second-order discretization logic has drifted."
    )

def test_gate_3a_gradient_finite_audit():
    """
    Verification: Fail-Fast audit for non-finite gradients.
    Compliance: Rule 7 (ArithmeticError trigger)
    """
    state = make_step2_output_dummy(nx=4, ny=4, nz=4)
    block = state.stencil_matrix[0]
    
    # Inject an instability
    block.i_plus.set_field(FI.P, np.nan)
    
    with pytest.raises(ArithmeticError):
        compute_local_gradient_p(block)