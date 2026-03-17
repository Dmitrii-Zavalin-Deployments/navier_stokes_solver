# tests/operators/test_forces.py

def test_get_body_forces_interior():
    nx, ny, nz = 5, 5, 5
    # The output should now be shape (3, nx-2, ny-2, nz-2)
    forces = get_body_forces_interior(nx, ny, nz, 0.0, -9.81, 0.0)
    
    assert forces.shape == (3, 3, 3, 3) 
    assert np.all(forces[1] == -9.81)