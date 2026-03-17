from src.step1.orchestrate_step1 import orchestrate_step1
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy


def test_staggered_face_volume_audit():
    """
    Gate Stage 1.A: Staggering Audit.
    Verifies Arakawa C-grid topology:
    - U-velocity: (nx+1, ny, nz) -> X-faces
    - V-velocity: (nx, ny+1, nz) -> Y-faces
    - W-velocity: (nx, ny, nz+1) -> Z-faces
    - Pressure: (nx, ny, nz)      -> Cell Centers
    """
    # 1. Setup
    input_data = solver_input_schema_dummy()
    nx, ny, nz = input_data["grid"]["nx"], input_data["grid"]["ny"], input_data["grid"]["nz"]
    
    # 2. Execution
    state = orchestrate_step1(input_data)
    
    # 3. Mathematical Check (Audit B.1)
    u_shape = state.fields["U"].shape
    v_shape = state.fields["V"].shape
    w_shape = state.fields["W"].shape
    p_shape = state.fields["P"].shape

    # U-velocity audit: (nx+1) faces
    assert u_shape == (nx + 1, ny, nz), f"U shape mismatch. Expected {(nx+1, ny, nz)}, got {u_shape}"
    
    # V-velocity audit: (ny+1) faces
    assert v_shape == (nx, ny + 1, nz), f"V shape mismatch. Expected {(nx, ny+1, nz)}, got {v_shape}"
    
    # W-velocity audit: (nz+1) faces
    assert w_shape == (nx, ny, nz + 1), f"W shape mismatch. Expected {(nx, ny, nz+1)}, got {w_shape}"
    
    # P-volume audit: (nx, ny, nz) centers
    assert p_shape == (nx, ny, nz), f"P shape mismatch. Expected {(nx, ny, nz)}, got {p_shape}"

    print(f"\n[STAGGERING AUDIT PASSED]: U:{u_shape}, V:{v_shape}, W:{w_shape}, P:{p_shape}")
