# tests/property_integrity/test_physics_fluid_constants.py

import numpy as np
import pytest

from tests.helpers.solver_output_schema_dummy import make_output_schema_dummy
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy

# Density and Viscosity are required starting from Step 3 (Projection & Diffusion)
PHYSICS_ACTIVE_STAGES = [
    ("Step 3", make_step3_output_dummy),
    ("Step 4", make_step4_output_dummy),
    ("Final Output", make_output_schema_dummy),
]

@pytest.mark.parametrize("stage_name, factory", PHYSICS_ACTIVE_STAGES)
def test_density_persistence_and_validity(stage_name, factory):
    """
    Physics: Verify that density (rho) is present and strictly positive 
    from Step 3 onwards to support the Pressure-Velocity correction logic.
    """
    state = factory()
    
    # 1. Existence Check
    assert hasattr(state.config, "density"), f"{stage_name}: Density missing from fluid_properties"
    
    # 2. Scale Guard: Density must be positive to prevent division by zero in correction
    rho = state.config.density
    assert rho > 0, f"{stage_name}: Non-physical density detected ({rho})"
    
    # 3. Type Check
    assert isinstance(rho, (float, int, np.float64)), f"{stage_name}: Density must be a numeric scalar"

@pytest.mark.parametrize("stage_name, factory", PHYSICS_ACTIVE_STAGES)
def test_viscosity_persistence_and_validity(stage_name, factory):
    """
    Physics: Verify that viscosity (nu) is present and strictly positive 
    from Step 3 onwards to support Momentum Diffusion logic.
    """
    state = factory()
    
    # 1. Existence Check
    assert hasattr(state.config, "viscosity"), f"{stage_name}: Viscosity missing from fluid_properties"
    
    # 2. Scale Guard: Negative viscosity is mathematically unstable (anti-diffusion)
    nu = state.config.viscosity
    assert nu > 0, f"{stage_name}: Non-physical viscosity detected ({nu})"
    
    # 3. Type Check
    assert isinstance(nu, (float, int, np.float64)), f"{stage_name}: Viscosity must be a numeric scalar"

def test_velocity_correction_scaling_consistency():
    """
    Theory: Ensure that the relationship dt/rho is calculable.
    This is the coefficient used in Step 3: u_new = u_star - (dt/rho) * grad(P)
    """
    state = make_step3_output_dummy()
    
    dt = state.config.time_step
    rho = state.config.density
    
    correction_factor = dt / rho
    assert np.isfinite(correction_factor), "Step 3: Velocity correction factor (dt/rho) is invalid"

def test_diffusion_stability_coefficient():
    """
    Theory: Ensure the diffusion coefficient (nu * dt / dx^2) is calculable.
    This relates to the stability of the explicit diffusion step.
    """
    state = make_step3_output_dummy()
    
    nu = state.config.viscosity
    dt = state.config.time_step
    dx = state.grid.dx
    
    # Stability factor must be a finite number
    stability_factor = (nu * dt) / (dx**2)
    assert np.isfinite(stability_factor), "Step 3: Diffusion stability factor is invalid"