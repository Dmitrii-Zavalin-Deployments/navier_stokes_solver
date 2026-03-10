# tests/helpers/solver_input_schema_dummy.py

from src.common.solver_input import SolverInput

def solver_input_schema_dummy() -> dict:
    """
    Returns a raw dictionary representing valid JSON input.
    Compliant with Phase C: Deterministic Initialization and Law of Singular Access.
    """
    nx, ny, nz = 2, 2, 2
    # Rule 5: Explicit initialization. 
    # Mask values: 1 = fluid, 0 = solid, -1 = boundary-fluid
    mask_flat = [1] * (nx * ny * nz) 

    return {
        "domain_configuration": {
            "type": "INTERNAL",
            "reference_velocity": [0.0, 0.0, 0.0]
        },
        "grid": {
            "x_min": 0.0, "x_max": 1.0,
            "y_min": 0.0, "y_max": 1.0,
            "z_min": 0.0, "z_max": 1.0,
            "nx": nx, "ny": ny, "nz": nz,
        },
        "fluid_properties": {
            "density": 1.0,
            "viscosity": 0.001,
        },
        "initial_conditions": {
            "velocity": [0.0, 0.0, 0.0],
            "pressure": 0.0,
        },
        "simulation_parameters": {
            "time_step": 0.01,
            "total_time": 0.1,
            "output_interval": 1,
        },
        "boundary_conditions": [
            {"location": "x_min", "type": "inflow", "values": {"u": 1.0, "v": 0.0, "w": 0.0}},
            {"location": "x_max", "type": "outflow", "values": {"p": 0.0}},
            {"location": "y_min", "type": "no-slip", "values": {"u": 0.0, "v": 0.0, "w": 0.0}},
            {"location": "y_max", "type": "free-slip", "values": {"u": 1.0, "w": 0.0}},
            {"location": "z_min", "type": "pressure", "values": {"p": 101325.0}},
            {"location": "z_max", "type": "pressure", "values": {"p": 0.0}},
            {"location": "wall", "type": "no-slip", "values": {"u": 0.0, "v": 0.0, "w": 0.0}},
        ],
        "mask": mask_flat,
        "external_forces": {
            "force_vector": [0.0, -9.81, 0.0]
        },
    }

def make_solver_input_dummy() -> SolverInput:
    """
    Returns a fully hydrated SolverInput OBJECT.
    Rule 8: Singular access; all data must pass through .from_dict().
    """
    return SolverInput.from_dict(solver_input_schema_dummy())