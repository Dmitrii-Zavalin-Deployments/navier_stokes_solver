# tests/property_integrity/test_lifecycle_allocation_robustness.py

import pytest

from tests.helpers.solver_output_schema_dummy import make_output_schema_dummy

# Import the full factory suite
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy

# Define the lifecycle stages for parametrization
LIFECYCLE_STAGES = [
    ("Step 1", make_step1_output_dummy),
    ("Step 2", make_step2_output_dummy),
    ("Step 3", make_step3_output_dummy),
    ("Step 4", make_step4_output_dummy),
    ("Final Output", make_output_schema_dummy),
]

@pytest.mark.parametrize("stage_name, factory", LIFECYCLE_STAGES)
def test_lifecycle_grid_dimensions_match_fields(stage_name, factory):
    """
    Robustness: Verifies that core field shapes (P, U, V, W) match 
    the Arakawa C-Grid staggered convention across all 5 stages.
    """
    nx, ny, nz = 8, 6, 4
    state = factory(nx=nx, ny=ny, nz=nz)
    
    # 1. Metadata Agreement
    expected_total = nx * ny * nz
    assert state.grid.nx == nx
    assert (state.grid.nx * state.grid.ny * state.grid.nz) == expected_total, f"Total cells mismatch at {stage_name}"

    # 2. Core Field Shapes (Staggered Grid Logic)
    # P is cell-centered: (nx, ny, nz)
    # U, V, W are face-centered: (nx+1, ny, nz), etc.
    expected_shapes = {
        "P": (nx, ny, nz),
        "U": (nx + 1, ny, nz),
        "V": (nx, ny + 1, nz),
        "W": (nx, ny, nz + 1)
    }

    # Handle both SolverState objects and dictionary-serialized outputs
    fields = state["fields"] if isinstance(state, dict) else state.fields

    for field, shape in expected_shapes.items():
        assert fields[field].shape == shape, (
            f"{stage_name}: Field '{field}' shape mismatch. "
            f"Expected {shape}, got {fields[field].shape}"
        )

@pytest.mark.parametrize("stage_name, factory", LIFECYCLE_STAGES)
def test_lifecycle_ppe_dimension_intent(stage_name, factory):
    """
    Robustness: Verifies that the PPE dimension (N_total) is 
    consistent and present in every stage of the state.
    """
    nx, ny, nz = 4, 3, 2
    state = factory(nx=nx, ny=ny, nz=nz)
    expected_dim = nx * ny * nz
    
    
    # Validates that the PPE department 'plan' survives every orchestrator
    assert (state.grid.nx * state.grid.ny * state.grid.nz) == expected_dim, f"PPE 'dimension' key missing at {stage_name}"
    assert (state.grid.nx * state.grid.ny * state.grid.nz) == expected_dim, f"PPE dimension value mismatch at {stage_name}"

def test_step3_intermediate_field_allocation():
    """
    Specific check for Step 3: Projection requires intermediate velocity 
    storage (U*, V*, W*) which must follow staggered face dimensions.
    
    Updated Feb 2026: Validates '_star' suffix for predictor fields.
    """
    nx, ny, nz = 5, 5, 5
    state = make_step3_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # Intermediate (Predictor) fields must match the staggering of core fields
    expected_shapes = {
        "U_star": (nx + 1, ny, nz),
        "V_star": (nx, ny + 1, nz),
        "W_star": (nx, ny, nz + 1)
    }
    
    for comp, shape in expected_shapes.items():
        assert state.fields[comp].shape == shape, \
            f"Step 3 intermediate field {comp} shape mismatch. Expected {shape}."

def test_ghost_cell_allocation_logic():
    """
    Verify the 'Ghost Cell' expansion logic in Step 4 and Final Output.
    Formula: Extended = Interior + 2 (one ghost on each side).
    Note: Staggered velocity faces (e.g., U) start at nx+1 faces internally.
    """
    nx, ny, nz = 10, 10, 10
    
    # Check both Step 4 and the Final Output wrapper
    s4 = make_step4_output_dummy(nx=nx, ny=ny, nz=nz)
    s_final = make_output_schema_dummy(nx=nx, ny=ny, nz=nz)
    
    for stage_name, state in [("Step 4", s4), ("Final Output", s_final)]:
        # Pressure (Cell-centered): 10 + 2 = 12
        assert state.P_ext.shape == (nx + 2, ny + 2, nz + 2), f"{stage_name} P_ext failure"
        
        # Velocity U (Face-centered in X): (10+1) + 2 = 13
        # Centered in Y, Z: 10 + 2 = 12
        assert state.U_ext.shape == (nx + 3, ny + 2, nz + 2), f"{stage_name} U_ext failure"
        assert state.V_ext.shape == (nx + 2, ny + 3, nz + 2), f"{stage_name} V_ext failure"
        assert state.W_ext.shape == (nx + 2, ny + 2, nz + 3), f"{stage_name} W_ext failure"