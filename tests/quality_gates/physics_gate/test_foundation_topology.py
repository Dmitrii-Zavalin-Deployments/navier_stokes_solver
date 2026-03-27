# tests/quality_gates/physics_gate/test_foundation_topology.py

from src.step1.orchestrate_step1 import orchestrate_step1
from src.step2.stencil_assembler import assemble_stencil_matrix


def test_gate_1a_foundation_volume_parity(solver_input_schema_dummy):
    """
    Gate 1.A: Foundation Volume Audit
    Identity: V_total = (nx+2)(ny+2)(nz+2)
    Compliance: Rule 9 (Hybrid Memory Foundation allocation)
    """
    context = solver_input_schema_dummy
    nx, ny, nz = context.input_data.grid.nx, context.input_data.grid.ny, context.input_data.grid.nz
    
    # The Foundation must allocate space for the core + 2 ghost layers per dimension
    expected_volume = (nx + 2) * (ny + 2) * (nz + 2)
    
    # Action: Orchestrate Step 1
    state = orchestrate_step1(context)
    
    # Verification: Access FieldManager.data (The Foundation Buffer)
    actual_volume = state.fields.data.shape[0]
    assert actual_volume == expected_volume, (
        f"Foundation Volume Breach: {actual_volume} allocated, "
        f"but math requires {expected_volume} for ghost padding."
    )

def test_gate_2a_7_point_connectivity_stride(solver_input_schema_dummy):
    """
    Gate 2.A: Topology Connectivity Audit
    Identity: id(i+1) = id(c) + stride_x
    Compliance: Rule 7 (Deterministic Indexing via get_flat_index)
    """
    # Action: Initialize full topology
    state = orchestrate_step1(solver_input_schema_dummy)
    state.stencil_matrix = assemble_stencil_matrix(state)
    
    # In a flattened array with order (z, y, x), stride_x is always 1.
    # Note: If your get_flat_index uses a different axis order, update stride_x accordingly.
    stride_x = 1 
    
    # Grab a block from the logical core (e.g., the first physical cell)
    # The assembler loops range(0, nx), so index 0 is logical (0,0,0)
    block = state.stencil_matrix[0]
    
    # Verification: Check connectivity pointers
    center_idx = block.center.index
    ip_idx = block.i_plus.index
    im_idx = block.i_minus.index
    
    assert ip_idx == center_idx + stride_x, (
        f"Topology Breach [X+]: Center index {center_idx} to i_plus {ip_idx} "
        f"does not follow stride {stride_x}."
    )
    assert im_idx == center_idx - stride_x, (
        f"Topology Breach [X-]: Center index {center_idx} to i_minus {im_idx} "
        f"does not follow stride -{stride_x}."
    )

def test_gate_2a_y_stride_integrity(solver_input_schema_dummy):
    """
    Gate 2.A Supplemental: Verify Y-axis stride logic
    Identity: id(j+1) = id(c) + (nx+2)
    """
    state = orchestrate_step1(solver_input_schema_dummy)
    state.stencil_matrix = assemble_stencil_matrix(state)
    
    nx_total = state.grid.nx + 2
    stride_y = nx_total
    
    # Grab an interior block
    block = state.stencil_matrix[0]
    
    center_idx = block.center.index
    jp_idx = block.j_plus.index
    
    assert jp_idx == center_idx + stride_y, (
        f"Topology Breach [Y+]: Center index {center_idx} to j_plus {jp_idx} "
        f"expected stride {stride_y} (nx+2)."
    )