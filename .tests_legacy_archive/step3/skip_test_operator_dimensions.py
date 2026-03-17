from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


def test_staggered_operator_contract():
    """
    PROVE: Sparse operators (lap_u, advection_u, etc.) match staggered field sizes.
    Equation: Output = Operator @ Flattened_Field
    The number of columns in the Operator must equal the number of elements in the Field.
    """
    nx, ny, nz = 3, 3, 3
    state = make_step2_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # Define the expected mapping: Field Name -> Operator Key
    # U-staggered: (nx+1, ny, nz) -> 4 * 3 * 3 = 36 elements
    # V-staggered: (nx, ny+1, nz) -> 3 * 4 * 3 = 36 elements
    # W-staggered: (nx, ny, nz+1) -> 3 * 3 * 4 = 36 elements
    
    checks = [
        ("U", ["lap_u", "advection_u"]),
        ("V", ["lap_v", "advection_v"]),
        ("W", ["lap_w", "advection_w"])
    ]
    
    mismatches = []
    
    for field_key, op_keys in checks:
        field_size = state.fields[field_key].size # Flattened count
        
        for op_key in op_keys:
            op = state.operators.get(op_key)
            if op is None:
                continue # Skip if operator wasn't implemented yet
            
            # Sparse matrix shape is (Rows, Cols)
            # Cols must match Input size; Rows must match Output size
            rows, cols = op.shape
            
            if cols != field_size:
                mismatches.append(
                    f"Operator '{op_key}' (cols={cols}) does not match '{field_key}' (size={field_size})"
                )
            if rows != field_size:
                mismatches.append(
                    f"Operator '{op_key}' (rows={rows}) would change size of '{field_key}' (expected={field_size})"
                )

    assert not mismatches, "Operator shape violations found:\n" + "\n".join(mismatches)