# tests/property_integrity/test_architecture_parity.py

import pytest

from src.common.field_schema import FI
from src.common.stencil_block import StencilBlock
from tests.helpers.solver_output_schema_dummy import make_output_schema_dummy

# Dummies updated to match the orchestration steps in main_solver.py
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy

# --- ARCHITECTURE CATEGORIZATION ---

# Stages using the monolithic SolverState.fields manager (per src/main_solver.py)
STATE_BASED_STAGES = [
    ("Step 1 (Init)", make_step1_output_dummy),
    ("Step 2 (Assembly)", make_step2_output_dummy),
    ("Step 4 (Archivist)", make_step4_output_dummy),
    ("Final Output", make_output_schema_dummy),
]

# Stages using component-based StencilBlocks (Inside the Step 3 loop)
BLOCK_BASED_STAGES = [
    ("Step 3 (PPE/Predictor)", make_step3_output_dummy),
]

ALL_STAGES = STATE_BASED_STAGES + BLOCK_BASED_STAGES

# --- ARCHITECTURE BRIDGE HELPERS ---

def get_fluid_param(obj, param_name):
    """Extracts physics constants regardless of the container type."""
    if isinstance(obj, StencilBlock):
        # StencilBlocks use direct private attributes for density/viscosity
        mapping = {"density": "_rho", "viscosity": "_mu"}
        return getattr(obj, mapping[param_name], None)
    
    # SolverState uses the _fluid_properties object
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
    n_expected = (nx + 2) * (ny + 2) * (nz + 2)
    state = factory(nx=nx, ny=ny, nz=nz)
    
    # Verify the monolithic fields.data array
    data = state.fields.data
    for field_idx in [FI.P, FI.VX, FI.VY, FI.VZ]:
        actual_size = data[:, field_idx].size
        assert actual_size == n_expected, (
            f"{stage_name}: Field {field_idx} size mismatch. "
            f"Expected {n_expected}, got {actual_size}"
        )

@pytest.mark.parametrize("stage_name, factory", BLOCK_BASED_STAGES)
def test_block_allocation_integrity(stage_name, factory):
    """
    Validation: Verify StencilBlocks are wired to the global Foundation buffer.
    In Step 3, Cell attributes (vx, vy, etc.) act as views into the monolithic array.
    """
    nx, ny, nz = 5, 5, 5
    # The total volume including the +2 ghost padding: (5+2)^3 = 343
    n_expected = (nx + 2) * (ny + 2) * (nz + 2)
    
    block = factory(nx=nx, ny=ny, nz=nz)
    
    # Audit each physical field for wiring integrity
    for attr in ["vx", "vy", "vz", "p"]:
        val = getattr(block.center, attr)
        
        assert val is not None, f"{stage_name}: Component {attr} is None"
        
        # GATE 1: The "View" Integrity (Object Layer)
        # Each Cell attribute must be a single-element view (size 1)
        assert val.size == 1, (
            f"{stage_name}: {attr} should be a single-element view, "
            f"but detected size {val.size}."
        )

        # GATE 2: The "Foundation" Integrity (NumPy Layer)
        # We check '.base' to ensure it is wired to the global [Cells x Fields] buffer.
        assert hasattr(val, "base") and val.base is not None, (
            f"{stage_name}: {attr} is a detached copy (Rule 9 Violation). "
            "Memory must be a VIEW of the global Foundation buffer."
        )
        
        # Calculation: 343 cells * 9 fields = 3087
        n_total_foundation = n_expected * FI.num_fields()
        actual_buffer_size = val.base.size
        
        assert actual_buffer_size == n_total_foundation, (
            f"{stage_name}: {attr} wiring mismatch. "
            f"Expected global buffer of {n_total_foundation}, "
            f"but detected {actual_buffer_size}."
        )

# --- PHYSICS & BOUNDARY PERSISTENCE ---

@pytest.mark.parametrize("stage_name, factory", ALL_STAGES)
def test_fluid_constants_persistence(stage_name, factory):
    """Physics: Ensure density and viscosity are positive and reachable in all stages."""
    obj = factory()
    rho = get_fluid_param(obj, "density")
    mu = get_fluid_param(obj, "viscosity")
    
    assert rho is not None and rho > 0, f"{stage_name}: Missing or non-physical density: {rho}"
    assert mu is not None and mu > 0, f"{stage_name}: Missing or non-physical viscosity: {mu}"

def test_staggered_component_schema_validity():
    """Safety: Verify BC values strictly follow the component schema: {u, v, w, p}."""
    # We check Step 1 as it is the entry point for BC definition
    state = make_step1_output_dummy()
    allowed_keys = {"u", "v", "w", "p"}
    bcs = get_bc_list(state)
    
    assert len(bcs) > 0, "Test Setup Error: Dummy state has no boundary conditions to audit."
    
    for bc in bcs:
        provided_keys = set(getattr(bc, "_values", {}).keys())
        # Ensure no illegal keys (like 'velocity_x') leaked into the dictionary
        assert provided_keys.issubset(allowed_keys), (
            f"Illegal component in BC: {provided_keys - allowed_keys}. "
            f"Must be subset of {allowed_keys}"
        )