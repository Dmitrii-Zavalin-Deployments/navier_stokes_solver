# src/solver_input.py

from dataclasses import dataclass, field

from src.common.base_container import ValidatedContainer

# =========================================================
# 1. SUB-DEPARTMENT CONTAINERS
# =========================================================

@dataclass
class GridInput(ValidatedContainer):
    _x_min: float = None; _x_max: float = None
    _y_min: float = None; _y_max: float = None
    _z_min: float = None; _z_max: float = None
    _nx: int = None; _ny: int = None; _nz: int = None

    # --- Properties (Getters/Setters) ---
    @property
    def x_min(self) -> float: return self._get_safe("x_min")
    @x_min.setter
    def x_min(self, v: float): self._set_safe("x_min", v, float)

    @property
    def x_max(self) -> float: return self._get_safe("x_max")
    @x_max.setter
    def x_max(self, v: float): self._set_safe("x_max", v, float)

    @property
    def nx(self) -> int: return self._get_safe("nx")
    @nx.setter
    def nx(self, v: int): 
        if v is not None and v < 1: raise ValueError(f"nx must be >= 1, got {v}")
        self._set_safe("nx", v, int)

    @property
    def y_min(self) -> float: return self._get_safe("y_min")
    @y_min.setter
    def y_min(self, v: float): self._set_safe("y_min", v, float)

    @property
    def y_max(self) -> float: return self._get_safe("y_max")
    @y_max.setter
    def y_max(self, v: float): self._set_safe("y_max", v, float)

    @property
    def ny(self) -> int: return self._get_safe("ny")
    @ny.setter
    def ny(self, v: int): 
        if v is not None and v < 1: raise ValueError(f"ny must be >= 1, got {v}")
        self._set_safe("ny", v, int)

    @property
    def z_min(self) -> float: return self._get_safe("z_min")
    @z_min.setter
    def z_min(self, v: float): self._set_safe("z_min", v, float)

    @property
    def z_max(self) -> float: return self._get_safe("z_max")
    @z_max.setter
    def z_max(self, v: float): self._set_safe("z_max", v, float)

    @property
    def nz(self) -> int: return self._get_safe("nz")
    @nz.setter
    def nz(self, v: int): 
        if v is not None and v < 1: raise ValueError(f"nz must be >= 1, got {v}")
        self._set_safe("nz", v, int)

    # --- Geometric Derived Properties (Spatial Governor) ---
    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / self.nx

    @property
    def dy(self) -> float:
        return (self.y_max - self.y_min) / self.ny

    @property
    def dz(self) -> float:
        return (self.z_max - self.z_min) / self.nz

    @property
    def total_cells(self) -> int:
        return self.nx * self.ny * self.nz

@dataclass
class FluidInput(ValidatedContainer):
    _density: float = None
    _viscosity: float = None

    @property
    def density(self) -> float: return self._get_safe("density")
    @density.setter
    def density(self, v: float):
        if v is not None and v <= 0: raise ValueError(f"Density must be > 0, got {v}")
        self._set_safe("density", v, float)

    @property
    def viscosity(self) -> float: return self._get_safe("viscosity")
    @viscosity.setter
    def viscosity(self, v: float):
        if v is not None and v < 0: raise ValueError(f"Viscosity must be >= 0, got {v}")
        self._set_safe("viscosity", v, float)

@dataclass
class InitialConditionsInput(ValidatedContainer):
    _velocity: list = None
    _pressure: float = None

    @property
    def velocity(self) -> list: return self._get_safe("velocity")
    @velocity.setter
    def velocity(self, v: list):
        if v is not None and len(v) != 3: raise ValueError("velocity must have 3 items")
        self._set_safe("velocity", v, list)

    @property
    def pressure(self) -> float: return self._get_safe("pressure")
    @pressure.setter
    def pressure(self, v: float): self._set_safe("pressure", v, float)

@dataclass
class SimParamsInput(ValidatedContainer):
    _time_step: float = None
    _total_time: float = None
    _output_interval: int = None
    _advection_weight_base: float = None

    @property
    def time_step(self) -> float: return self._get_safe("time_step")
    @time_step.setter
    def time_step(self, v: float):
        if v is not None and v <= 0: raise ValueError("time_step must be > 0")
        self._set_safe("time_step", v, float)

    @property
    def total_time(self) -> float: return self._get_safe("total_time")
    @total_time.setter
    def total_time(self, v: float):
        if v is not None and v <= 0: raise ValueError("total_time must be > 0")
        self._set_safe("total_time", v, float)

    @property
    def output_interval(self) -> int: return self._get_safe("output_interval")
    @output_interval.setter
    def output_interval(self, v: int):
        if v is not None and v < 1: raise ValueError("output_interval must be >= 1")
        self._set_safe("output_interval", v, int)

    @property
    def advection_weight_base(self) -> float: 
        return self._get_safe("advection_weight_base")
        
    @advection_weight_base.setter
    def advection_weight_base(self, v: float):
        if v is not None:
            if v < 0: raise ValueError(f"advection_weight_base must be >= 0, got {v}")
            if v > 1: raise ValueError(f"advection_weight_base must be <= 1, got {v}")
        self._set_safe("advection_weight_base", v, float)

@dataclass
class BoundaryConditionItem(ValidatedContainer):
    _location: str = None 
    _type: str = None      
    _values: dict = field(default_factory=dict)
    _comment: str = ""

    @property
    def location(self) -> str: return self._get_safe("location")
    @location.setter
    def location(self, v: str):
        valid = ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]
        if v not in valid: raise ValueError(f"Invalid location: {v}")
        self._set_safe("location", v, str)

    @property
    def type(self) -> str: return self._get_safe("type")
    @type.setter
    def type(self, v: str):
        valid = ["no-slip", "free-slip", "inflow", "outflow", "pressure"]
        if v not in valid: raise ValueError(f"Invalid type: {v}")
        self._set_safe("type", v, str)

    @property
    def values(self) -> dict: return self._get_safe("values")
    @values.setter
    def values(self, v: dict): self._set_safe("values", v, dict)

    @property
    def comment(self) -> str: return self._get_safe("comment")
    @comment.setter
    def comment(self, v: str): self._set_safe("comment", v, str)

@dataclass
class BoundaryConditionsInput(ValidatedContainer):
    _items: list[BoundaryConditionItem] = field(default_factory=list)

    @property
    def items(self) -> list[BoundaryConditionItem]: return self._get_safe("items")
    @items.setter
    def items(self, v: list):
        processed = []
        if isinstance(v, list):
            for entry in v:
                if isinstance(entry, dict):
                    bc = BoundaryConditionItem()
                    bc.location = entry.get("location")
                    bc.type = entry.get("type")
                    bc.values = entry.get("values", {})
                    bc.comment = entry.get("comment", "")
                    processed.append(bc)
                else: processed.append(entry)
        self._set_safe("items", processed, list)

@dataclass
class MaskInput(ValidatedContainer):
    _data: list = None

    @property
    def data(self) -> list: return self._get_safe("data")
    @data.setter
    def data(self, v: list):
        if v is not None:
            if not all(val in {-1, 0, 1} for val in v):
                raise ValueError("Mask contains invalid values. Only -1, 0, 1 allowed.")
        self._set_safe("data", v, list)

@dataclass
class ExternalForcesInput(ValidatedContainer):
    _force_vector: list = None
    _comment: str = ""

    @property
    def force_vector(self) -> list: return self._get_safe("force_vector")
    @force_vector.setter
    def force_vector(self, v: list):
        if v is not None and len(v) != 3: raise ValueError("force_vector must have 3 items")
        self._set_safe("force_vector", v, list)

    @property
    def comment(self) -> str: return self._get_safe("comment")
    @comment.setter
    def comment(self, v: str): self._set_safe("comment", v, str)

# =========================================================
# 2. THE UNIVERSAL INPUT CONTAINER
# =========================================================

@dataclass
class SolverInput(ValidatedContainer):
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
        
        g = data.get("grid", {})
        for k in ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "nx", "ny", "nz"]:
            if k in g: setattr(obj.grid, k, g[k])
            
        f = data.get("fluid_properties", {})
        if "density" in f: obj.fluid_properties.density = f["density"]
        if "viscosity" in f: obj.fluid_properties.viscosity = f["viscosity"]
        
        ic = data.get("initial_conditions", {})
        if "velocity" in ic: obj.initial_conditions.velocity = ic["velocity"]
        if "pressure" in ic: obj.initial_conditions.pressure = ic["pressure"]
        
        sp = data.get("simulation_parameters", {})
        if "time_step" in sp: obj.simulation_parameters.time_step = sp["time_step"]
        if "total_time" in sp: obj.simulation_parameters.total_time = sp["total_time"]
        if "output_interval" in sp: obj.simulation_parameters.output_interval = sp["output_interval"]
        # NEW: Hydrate the advection weight
        if "advection_weight_base" in sp: 
            obj.simulation_parameters.advection_weight_base = sp["advection_weight_base"]
        
        ef = data.get("external_forces", {})
        if "force_vector" in ef: obj.external_forces.force_vector = ef["force_vector"]
        if "comment" in ef: obj.external_forces.comment = ef.get("comment", "")
        
        obj.mask.data = data.get("mask", [])
        obj.boundary_conditions.items = data.get("boundary_conditions", [])
        
        return obj

    def to_dict(self) -> dict:
        """Serializes back to a clean JSON-compatible dictionary."""
        def get_val(obj, key, default=None):
            if isinstance(obj, dict): return obj.get(key, default)
            return getattr(obj, key, default)

        return {
            "grid": {k: get_val(self.grid, k) for k in ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "nx", "ny", "nz"]},
            "fluid_properties": {k: get_val(self.fluid_properties, k) for k in ["density", "viscosity"]},
            "initial_conditions": {k: get_val(self.initial_conditions, k) for k in ["velocity", "pressure"]},
            "simulation_parameters": {
                k: get_val(self.simulation_parameters, k) 
                # UPDATED: Included advection_weight_base in the keys list
                for k in ["time_step", "total_time", "output_interval", "advection_weight_base"]
            },
            "boundary_conditions": [
                {k: get_val(bc, k) for k in ["location", "type", "values", "comment"]}
                for bc in self.boundary_conditions.items
            ],
            "mask": self.mask.data,
            "external_forces": {k: get_val(self.external_forces, k) for k in ["force_vector", "comment"]}
        }