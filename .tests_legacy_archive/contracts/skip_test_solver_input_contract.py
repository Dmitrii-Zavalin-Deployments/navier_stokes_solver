# tests/contracts/test_solver_input_contract.py

import pytest

from src.solver_input import BoundaryConditionItem, SolverInput
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy


class TestSolverInputContract:
    """
    THE CONSTITUTIONAL PROXY: Verifies that SolverInput acts as a 
    perfect behavioral implementation of the JSON Schema.
    """

    def test_round_trip_integrity(self):
        """
        GIVEN a raw dictionary that matches the JSON Schema
        WHEN it is hydrated into SolverInput and then serialized back via to_dict()
        THEN the output should match the input 100%.
        """
        raw_input = solver_input_schema_dummy()
        obj = SolverInput.from_dict(raw_input)
        serialized_output = obj.to_dict()
        assert serialized_output == raw_input

    def test_boundary_conditions_object_injection(self):
        """
        HITS LINE 183-188: Ensures the setter accepts pre-instantiated 
        BoundaryConditionItem objects, not just dictionaries.
        """
        obj = SolverInput()
        manual_bc = BoundaryConditionItem()
        manual_bc.location = "x_min"
        manual_bc.type = "no-slip"
        
        # Injecting the object directly (triggers the 'else' branch in items setter)
        obj.boundary_conditions.items = [manual_bc]
        
        assert len(obj.boundary_conditions.items) == 1
        assert obj.boundary_conditions.items[0].location == "x_min"

    def test_schema_minimum_constraints(self):
        """Verifies 'minimum' and 'exclusiveMinimum' from schema."""
        obj = SolverInput()

        with pytest.raises(ValueError, match="nx must be >= 1"):
            obj.grid.nx = 0
        
        with pytest.raises(ValueError, match="Density must be > 0"):
            obj.fluid_properties.density = 0

        with pytest.raises(ValueError, match="Viscosity must be >= 0"):
            obj.fluid_properties.viscosity = -1.0

        with pytest.raises(ValueError, match="Mask contains invalid values"):
            obj.mask.data = [2, 0, -1]

    def test_schema_array_size_constraints(self):
        """Ensures 3D spatial vectors (velocity, forces) are strictly 3 items."""
        obj = SolverInput()

        with pytest.raises(ValueError, match="velocity must have 3 items"):
            obj.initial_conditions.velocity = [0.0, 0.0]

        with pytest.raises(ValueError, match="force_vector must have 3 items"):
            obj.external_forces.force_vector = [0.0, -9.8, 0.0, 1.0]

    def test_boundary_condition_enums(self):
        """Enforces the 'enum' constraints from the JSON schema."""
        obj = SolverInput()
        
        with pytest.raises(ValueError, match="Invalid location"):
            obj.boundary_conditions.items = [{"location": "center", "type": "no-slip"}]
            
        with pytest.raises(ValueError, match="Invalid type"):
            obj.boundary_conditions.items = [{"location": "x_min", "type": "warp-speed"}]

    def test_optional_comments_and_defaults(self):
        """Ensures preservation of metadata and default empty strings."""
        raw_input = solver_input_schema_dummy()
        raw_input["external_forces"]["comment"] = "Testing gravity"
        
        obj = SolverInput.from_dict(raw_input)
        
        # Check explicit comment
        assert obj.external_forces.comment == "Testing gravity"
        
        # Check default comment preservation (hits line 183 in Item factory)
        bc_item = obj.boundary_conditions.items[0]
        assert isinstance(bc_item.comment, str)
    
    def test_advection_weight_bounds(self):
        """Verifies the [0, 1] bounds for advection_weight_base."""
        obj = SolverInput()
        
        with pytest.raises(ValueError, match="advection_weight_base must be >= 0"):
            obj.simulation_parameters.advection_weight_base = -0.1
            
        with pytest.raises(ValueError, match="advection_weight_base must be <= 1"):
            obj.simulation_parameters.advection_weight_base = 1.1