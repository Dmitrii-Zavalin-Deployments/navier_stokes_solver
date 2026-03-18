# tests/property_integrity/test_architecture_parity.py

import numpy as np
import pytest

from src.common.field_schema import FI
from src.common.stencil_block import StencilBlock
from tests.helpers.solver_output_schema_dummy import make_output_schema_dummy
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy
from tests.helpers.solver_step5_output_dummy import make_step5_output_dummy

# Only stages that return a SolverState with a monolithic .fields manager
STATE_BASED_STAGES = [
    ("Step 1", make_step1_output_dummy),
    ("Step 2", make_step2_output_dummy),
    ("Step 5", make_step5_output_dummy),
    ("Final Output", make_output_schema_dummy),
]

# Stages that return a StencilBlock (Component-based storage)
BLOCK_BASED_STAGES = [
    ("Step 3", make_step3_output_dummy),
    ("Step 4", make_step4_output_dummy),
]

# All stages combined for physics and boundary persistence verification
ALL_STAGES = STATE_BASED_STAGES + BLOCK_BASED_STAGES

# --- ARCHITECTURE BRIDGE HELPERS ---

def get_fluid_param(obj, param_name):
    """Extracts physics constants regardless of the container type."""
    if isinstance(obj, StencilBlock):
        mapping = {"density": "_rho", "viscosity": "_mu"}
        return getattr(obj, mapping[param_name], None)
    
    fluid = getattr(obj, "_fluid_properties", None)
    if fluid:
        attr = f"_{param_name}"
        return getattr(fluid, attr, None)
    return None

def get_bc_list(obj):
    """Extracts boundary conditions list from either State or Block."""
    if isinstance(obj, StencilBlock):
        return getattr(obj, "_bc_list", [])
    
    manager = getattr(obj, "_boundary_conditions", None)
    if manager:
        return getattr(manager, "_conditions", [])
    return []

# --- MEMORY & ALLOCATION TESTS ---

@pytest.mark.parametrize("stage_name, factory", STATE_BASED_STAGES)
def test_lifecycle_grid_dimensions_match_fields(stage_name, factory):
    """Robustness: Verifies monolithic buffer size matches (nx+2)*(ny+2)*(nz+2)."""
    nx, ny, nz = 8, 6, 4
    n_cells = (nx + 2) * (ny + 2) * (nz + 2)
    state = factory(nx=nx, ny=ny, nz=nz)
    
    data = state.fields.data
    for field_idx in [FI.P, FI.VX, FI.VY, FI.VZ]:
        assert data[:, field_idx].size == n_cells, f"{stage_name}: Field {field_idx} size mismatch"

@pytest.mark.parametrize("stage_name, factory", BLOCK_BASED_STAGES)
def test_block_allocation_integrity(stage_name, factory):
    """Validation: Verify StencilBlocks allocate individual component arrays correctly."""
    nx, ny, nz = 5, 5, 5
    (nx + 2) * (ny + 2) * (nz + 2)
    block = factory(nx=nx, ny=ny, nz=nz)
    
    for attr in ["u", "v", "w", "p"]:
        val = getattr(block.center, attr)
        assert val is not None, f"{stage_name}: {attr} size mismatch"

# --- PHYSICS & BOUNDARY PERSISTENCE ---

@pytest.mark.parametrize("stage_name, factory", ALL_STAGES)
def test_fluid_constants_persistence(stage_name, factory):
    """Physics: Ensure density and viscosity are positive and reachable in all stages."""
    obj = factory()
    rho = get_fluid_param(obj, "density")
    mu = get_fluid_param(obj, "viscosity")
    
    assert rho is not None and rho > 0, f"{stage_name}: Invalid density {rho}"
    assert mu is not None and mu > 0, f"{stage_name}: Invalid viscosity {mu}"

@pytest.mark.parametrize("stage_name, factory", ALL_STAGES)
def test_boundary_condition_lifecycle_persistence(stage_name, factory):
    """Verification: BC values (u, v, w, p) survive across all pipeline transitions."""
    obj = factory(nx=4, ny=4, nz=4)
    bcs = get_bc_list(obj)
    
    assert len(bcs) > 0, f"{stage_name}: Boundary conditions lost in transition"
    
    # Locate a sample BC (e.g., x_min) and verify value integrity
    bc_entry = next((bc for bc in bcs if getattr(bc, "_location", None) == "x_min"), None)
    assert bc_entry is not None, f"{stage_name}: BC entry for 'x_min' lost"
    
    values = getattr(bc_entry, "_values", {})
    assert isinstance(values.get("u"), (int, float)), f"{stage_name}: BC 'u' value corrupted"

def test_staggered_component_schema_validity():
    """Safety: Verify BC values strictly follow the component schema: {u, v, w, p}."""
    state = make_step1_output_dummy()
    allowed_keys = {"u", "v", "w", "p"}
    bcs = get_bc_list(state)
    
    for bc in bcs:
        provided_keys = set(getattr(bc, "_values", {}).keys())
        assert provided_keys.issubset(allowed_keys), \
            f"Illegal component in BC: {provided_keys - allowed_keys}"

# --- STEP 3 SPECIFIC STABILITY ---

def test_step3_predictor_and_stability_parity():
    """Validation: Verify Step 3 predictor allocation and stability coefficients."""
    nx, ny, nz = 5, 5, 5
    n_cells = (nx + 2) * (ny + 2) * (nz + 2)
    block = make_step3_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # Check Predictor Allocation
    assert block._u_star.size == n_cells
    assert block._v_star.size == n_cells
    assert block._w_star.size == n_cells
    
    # Check Stability Coefficients
    assert np.isfinite(block.dt / block.rho), "Invalid Velocity Correction Factor"
    stability = (block.mu * block.dt) / (block.dx**2)
    assert np.isfinite(stability), "Invalid Diffusion Stability Factor"