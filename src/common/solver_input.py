# src/common/solver_input.py

from dataclasses import dataclass, field

from src.common.base_container import ValidatedContainer

# ... [Keep all your Sub-Department Containers: DomainConfigInput, GridInput, etc.] ...

@dataclass
class SolverInput(ValidatedContainer):
    __slots__ = ['domain_configuration', 'grid', 'fluid_properties', 'initial_conditions', 
                 'simulation_parameters', 'external_forces', 'mask', 'boundary_conditions']
    
    domain_configuration: DomainConfigInput = field(default_factory=DomainConfigInput)
    grid: GridInput = field(default_factory=GridInput)
    fluid_properties: FluidInput = field(default_factory=FluidInput)
    initial_conditions: InitialConditionsInput = field(default_factory=InitialConditionsInput)
    simulation_parameters: SimParamsInput = field(default_factory=SimParamsInput)
    external_forces: ExternalForcesInput = field(default_factory=ExternalForcesInput)
    mask: MaskInput = field(default_factory=MaskInput)
    boundary_conditions: BoundaryConditionsInput = field(default_factory=BoundaryConditionsInput)

    @classmethod
    def from_dict(cls, data: dict) -> "SolverInput":
        obj = cls()
        
        # Domain Configuration
        dc = data["domain_configuration"]
        obj.domain_configuration.type = dc["type"]
        obj.domain_configuration.reference_velocity = dc.get("reference_velocity")
        
        # Grid
        g = data["grid"]
        for k in ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "nx", "ny", "nz"]:
            setattr(obj.grid, k, g[k])
            
        # Fluid
        f = data["fluid_properties"]
        obj.fluid_properties.density = f["density"]
        obj.fluid_properties.viscosity = f["viscosity"]
        
        # Initial Conditions
        ic = data["initial_conditions"]
        obj.initial_conditions.velocity = ic["velocity"]
        obj.initial_conditions.pressure = ic["pressure"]
        
        # Simulation Parameters
        sp = data["simulation_parameters"]
        obj.simulation_parameters.time_step = sp["time_step"]
        obj.simulation_parameters.total_time = sp["total_time"]
        obj.simulation_parameters.output_interval = sp["output_interval"]
        
        # Forces, Mask, Boundary Conditions
        obj.external_forces.force_vector = data["external_forces"]["force_vector"]
        obj.mask.data = data["mask"]
        obj.boundary_conditions.items = data["boundary_conditions"]
        
        return obj

    def to_dict(self) -> dict:
        """Returns a dictionary strictly adhering to the input schema."""
        return {
            "domain_configuration": {
                "type": self.domain_configuration.type,
                "reference_velocity": self.domain_configuration.reference_velocity
            },
            "grid": {k: getattr(self.grid, k) for k in ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "nx", "ny", "nz"]},
            "fluid_properties": {"density": self.fluid_properties.density, "viscosity": self.fluid_properties.viscosity},
            "initial_conditions": {"velocity": self.initial_conditions.velocity, "pressure": self.initial_conditions.pressure},
            "simulation_parameters": {
                "time_step": self.simulation_parameters.time_step, 
                "total_time": self.simulation_parameters.total_time, 
                "output_interval": self.simulation_parameters.output_interval
            },
            "boundary_conditions": [{"location": bc.location, "type": bc.type, "values": bc.values} for bc in self.boundary_conditions.items],
            "mask": self.mask.data,
            "external_forces": {"force_vector": self.external_forces.force_vector}
        }