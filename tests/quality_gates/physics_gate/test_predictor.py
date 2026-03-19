# tests/quality_gates/physics_gate/test_predictor.py

import pytest

from src.common.field_schema import FI
from src.step3.predictor import compute_local_predictor_step
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


def setup_predictor_block(dt=1.0, rho=1.0, mu=1.0, dx=1.0):
    state = make_step2_output_dummy(nx=4, ny=4, nz=4)
    block = state.stencil_matrix[10] # Central core block

    # Standardize physics to unit values
    params = {
        '_dx': float(dx), '_dy': float(dx), '_dz': float(dx),
        '_dt': float(dt), '_rho': float(rho), '_mu': float(mu),
        '_f_vals': (0.0, 0.0, 0.0) # No body forces for this specific math check
    }
    for attr, val in params.items():
        object.__setattr__(block, attr, val)

    block.center.fields_buffer.fill(0.0) 
    return block

def test_predictor_teamwork_diffusion_only():
    """
    Verifies that v* = v_n + (dt/rho) * (mu * lap(v_n))
    Expectation: With mu=1, dt=1, rho=1, and lap=2.0 -> v* = 0 + 2.0 = 2.0
    """
    block = setup_predictor_block(mu=1.0, dt=1.0, rho=1.0, dx=1.0)
    
    # Create a local curvature in VX: (ip: 1.0, c: 0.0, im: 1.0)
    # lap = (1 - 2*0 + 1) / 1^2 = 2.0
    block.i_plus.set_field(FI.VX, 1.0)
    block.i_minus.set_field(FI.VX, 1.0)
    
    compute_local_predictor_step(block)
    
    # Calculation: v* = 0 + (1/1) * (1.0 * 2.0) = 2.0
    assert block.center.get_field(FI.VX_STAR) == pytest.approx(2.0)

def test_predictor_3d_logic_verification():
    """
    Verification of the 3D Predictor Equation:
    v* = v_n + (dt/rho) * [ (mu * lap) - (rho * adv) - grad_p ]
    """
    block = setup_predictor_block(mu=1.0, dt=1.0, rho=1.0, dx=1.0)
    
    # --- 1. SET CENTER VELOCITY ---
    block.center.set_field(FI.VX, 1.0)
    
    # --- 2. SET X-AXIS NEIGHBORS (The Flow Direction) ---
    # Creates a linear slope: grad = 1.0, lap = 0.0
    block.i_plus.set_field(FI.VX, 2.0)
    block.i_minus.set_field(FI.VX, 0.0)
    
    # --- 3. SET Y & Z NEIGHBORS (The Planar Symmetry) ---
    # To simulate 1D flow in a 3D solver, Y/Z neighbors must match the center!
    # This prevents the Laplacian from 'feeling' friction from the sides.
    for neighbor in [block.j_plus, block.j_minus, block.k_plus, block.k_minus]:
        neighbor.set_field(FI.VX, 1.0)
        
    # --- 4. SET PRESSURE GRADIENT ---
    # ip=2, im=0 -> grad_p = (2-0)/2 = 1.0
    block.i_plus.set_field(FI.P, 2.0)
    block.i_minus.set_field(FI.P, 0.0)

    # --- EXECUTION ---
    compute_local_predictor_step(block)
    
    # --- MATH AUDIT ---
    # Laplacian: (2 - 2*1 + 0)/1 + (1 - 2*1 + 1)/1 + (1 - 2*1 + 1)/1 = 0.0
    # Advection: u * du/dx = 1.0 * (2-0)/2 = 1.0
    # Grad P: (2-0)/2 = 1.0
    # V* = 1.0 + 1.0 * [ (1*0.0) - (1*1.0) - 1.0 ] = -1.0
    
    obtained = block.center.get_field(FI.VX_STAR)
    assert obtained == pytest.approx(-1.0), f"Expected -1.0, got {obtained}"