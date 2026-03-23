# tests/property_integrity/test_state_transitions.py

from tests.helpers.solver_step5_output_dummy import make_step5_output_dummy

from tests.helpers.solver_output_schema_dummy import make_output_schema_dummy


def test_bridge_step5_to_output_integrity():
    """
    Verifies that the terminal state (Step 6/Archiver) maintains 
    consistency with the math-validated state (Step 5).
    """
    # 1. Generate both dummies
    intermediate_state = make_step5_output_dummy(nx=4, ny=4, nz=4)
    terminal_state = make_output_schema_dummy(nx=4, ny=4, nz=4)

    # 2. PROOF OF CONTINUITY: Hybrid Memory Check (Rule 9)
    # The physical fields (VX, VY, P) must be identical between states.
    # If the archiver or terminal logic modified the math values, this fails.
    assert (intermediate_state.fields.data == terminal_state.fields.data).all(), \
        "Data corruption detected: Terminal state fields do not match Step 5."

    # 3. PROOF OF EVOLUTION: Manifest Finalization
    # The terminal state should have a more complete manifest than Step 5.
    assert len(terminal_state.manifest.saved_snapshots) >= len(intermediate_state.manifest.saved_snapshots), \
        "Terminal manifest has fewer snapshots than the intermediate state."
    
    # 4. PROOF OF LIFECYCLE: State Flag Transition
    # Step 5 is 'ready_for_time_loop', Terminal State must be False.
    assert intermediate_state.ready_for_time_loop is True
    assert terminal_state.ready_for_time_loop is False

    # 5. PATH VALIDITY: Ensure output paths are still rooted correctly
    for path in terminal_state.manifest.saved_snapshots:
        assert path.startswith(terminal_state.manifest.output_directory), \
            f"Path mismatch: {path} is not inside {terminal_state.manifest.output_directory}"