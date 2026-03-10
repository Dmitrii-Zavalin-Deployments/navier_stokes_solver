# src/common/solver_state.py

from dataclasses import dataclass

import numpy as np

from src.common.base_container import ValidatedContainer

# =========================================================
# THE DEPARTMENT SAFES (Memory-Hardened Managers)
# =========================================================

@dataclass
class DomainManager(ValidatedContainer):
    __slots__ = ['_type', '_reference_velocity']
    
    # Internal state initialized to None (enforcing Rule 5: Zero-Debt)
    _type: str = None
    _reference_velocity: np.ndarray = None

    @property
    def type(self) -> str:
        return self._get_safe("type")

    @type.setter
    def type(self, value: str):
        if value not in ["INTERNAL", "EXTERNAL"]:
            raise ValueError(f"Invalid domain type: {value}. Must be INTERNAL or EXTERNAL.")
        self._set_safe("type", value, str)

    @property
    def reference_velocity(self) -> np.ndarray:
        return self._get_safe("reference_velocity")

    @reference_velocity.setter
    def reference_velocity(self, value: np.ndarray):
        if value is not None and (not isinstance(value, np.ndarray) or value.size != 3):
            raise TypeError("reference_velocity must be a 3D NumPy array.")
        self._set_safe("reference_velocity", value, np.ndarray)

@dataclass
class GridManager(ValidatedContainer):
    __slots__ = [
        '_x_min', '_x_max', '_y_min', '_y_max', '_z_min', '_z_max', 
        '_nx', '_ny', '_nz'
    ]

    # Internal state initialized to None (Zero-Debt Policy)
    _x_min: float = None; _x_max: float = None
    _y_min: float = None; _y_max: float = None
    _z_min: float = None; _z_max: float = None
    _nx: int = None; _ny: int = None; _nz: int = None

    # --- Setters and Getters with Security Firewall ---
    
    @property
    def x_min(self) -> float: return self._get_safe("x_min")
    @x_min.setter
    def x_min(self, value: float): self._set_safe("x_min", value, float)

    @property
    def x_max(self) -> float: return self._get_safe("x_max")
    @x_max.setter
    def x_max(self, value: float): self._set_safe("x_max", value, float)

    @property
    def y_min(self) -> float: return self._get_safe("y_min")
    @y_min.setter
    def y_min(self, value: float): self._set_safe("y_min", value, float)

    @property
    def y_max(self) -> float: return self._get_safe("y_max")
    @y_max.setter
    def y_max(self, value: float): self._set_safe("y_max", value, float)

    @property
    def z_min(self) -> float: return self._get_safe("z_min")
    @z_min.setter
    def z_min(self, value: float): self._set_safe("z_min", value, float)

    @property
    def z_max(self) -> float: return self._get_safe("z_max")
    @z_max.setter
    def z_max(self, value: float): self._set_safe("z_max", value, float)

    @property
    def nx(self) -> int: return self._get_safe("nx")
    @nx.setter
    def nx(self, value: int):
        if value < 1: raise ValueError("nx must be >= 1")
        self._set_safe("nx", value, int)

    @property
    def ny(self) -> int: return self._get_safe("ny")
    @ny.setter
    def ny(self, value: int):
        if value < 1: raise ValueError("ny must be >= 1")
        self._set_safe("ny", value, int)

    @property
    def nz(self) -> int: return self._get_safe("nz")
    @nz.setter
    def nz(self, value: int):
        if value < 1: raise ValueError("nz must be >= 1")
        self._set_safe("nz", value, int)

    # --- Derived Spacing Properties ---
    @property
    def dx(self) -> float: return (self.x_max - self.x_min) / self.nx
    @property
    def dy(self) -> float: return (self.y_max - self.y_min) / self.ny
    @property
    def dz(self) -> float: return (self.z_max - self.z_min) / self.nz

@dataclass
class FluidPropertiesManager(ValidatedContainer):
    __slots__ = ['_density', '_viscosity']
    
    # Internal state initialized to None (Zero-Debt Policy)
    _density: float = None
    _viscosity: float = None

    # --- Setters and Getters with Security Firewall ---

    @property
    def density(self) -> float: 
        return self._get_safe("density")

    @density.setter
    def density(self, value: float):
        if value is not None and value <= 0:
            raise ValueError(f"Density must be > 0, got {value}.")
        self._set_safe("density", value, float)

    @property
    def viscosity(self) -> float: 
        return self._get_safe("viscosity")

    @viscosity.setter
    def viscosity(self, value: float):
        if value is not None and value < 0:
            raise ValueError(f"Viscosity must be >= 0, got {value}.")
        self._set_safe("viscosity", value, float)

@dataclass
class InitialConditionManager(ValidatedContainer):
    __slots__ = ['_velocity', '_pressure']
    
    # Internal state initialized to None (Zero-Debt Policy)
    _velocity: np.ndarray = None
    _pressure: float = None

    # --- Setters and Getters with Security Firewall ---

    @property
    def velocity(self) -> np.ndarray:
        return self._get_safe("velocity")

    @velocity.setter
    def velocity(self, value: np.ndarray):
        # Validate 3D vector constraint
        if value is not None and (not isinstance(value, np.ndarray) or value.size != 3):
            raise ValueError(f"Velocity must be a 3D NumPy array, got {type(value)} with size {getattr(value, 'size', 'N/A')}.")
        self._set_safe("velocity", value, np.ndarray)

    @property
    def pressure(self) -> float:
        return self._get_safe("pressure")

    @pressure.setter
    def pressure(self, value: float):
        # Enforce type check via _set_safe
        self._set_safe("pressure", value, (float, int))

@dataclass
class SimulationParameterManager(ValidatedContainer):
    __slots__ = ['_time_step', '_total_time', '_output_interval']
    
    # Internal state initialized to None (Zero-Debt Policy)
    _time_step: float = None
    _total_time: float = None
    _output_interval: int = None

    # --- Setters and Getters with Security Firewall ---

    @property
    def time_step(self) -> float:
        return self._get_safe("time_step")

    @time_step.setter
    def time_step(self, value: float):
        if value is not None and value <= 0:
            raise ValueError(f"time_step must be > 0, got {value}.")
        self._set_safe("time_step", value, (float, int))

    @property
    def total_time(self) -> float:
        return self._get_safe("total_time")

    @total_time.setter
    def total_time(self, value: float):
        if value is not None and value <= 0:
            raise ValueError(f"total_time must be > 0, got {value}.")
        self._set_safe("total_time", value, (float, int))

    @property
    def output_interval(self) -> int:
        return self._get_safe("output_interval")

    @output_interval.setter
    def output_interval(self, value: int):
        if value is not None and value < 1:
            raise ValueError(f"output_interval must be >= 1, got {value}.")
        self._set_safe("output_interval", value, int)

@dataclass
class BoundaryCondition(ValidatedContainer):
    __slots__ = ['_location', '_type', '_values']
    
    # Internal state initialized to None (Zero-Debt Policy)
    _location: str = None
    _type: str = None
    _values: dict = None

    # --- Setters and Getters with Security Firewall ---

    @property
    def location(self) -> str:
        return self._get_safe("location")

    @location.setter
    def location(self, value: str):
        valid = ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "wall"]
        if value is not None and value not in valid:
            raise ValueError(f"Invalid location '{value}'. Must be one of {valid}.")
        self._set_safe("location", value, str)

    @property
    def type(self) -> str:
        return self._get_safe("type")

    @type.setter
    def type(self, value: str):
        valid = ["no-slip", "free-slip", "inflow", "outflow", "pressure"]
        if value is not None and value not in valid:
            raise ValueError(f"Invalid type '{value}'. Must be one of {valid}.")
        self._set_safe("type", value, str)

    @property
    def values(self) -> dict:
        return self._get_safe("values")

    @values.setter
    def values(self, value: dict):
        if value is not None and not isinstance(value, dict):
            raise TypeError("values must be a dictionary.")
        self._set_safe("values", value, dict)

@dataclass
class BoundaryConditionManager(ValidatedContainer):
    __slots__ = ['_conditions']
    
    # Internal state initialized to None (Zero-Debt Policy)
    _conditions: list[BoundaryCondition] = None

    # --- Setters and Getters with Security Firewall ---

    @property
    def conditions(self) -> list[BoundaryCondition]:
        return self._get_safe("conditions")

    @conditions.setter
    def conditions(self, value: list[BoundaryCondition]):
        if value is not None:
            if not isinstance(value, list):
                raise TypeError("conditions must be a list of BoundaryCondition objects.")
            # Ensure every item in the list is a valid BoundaryCondition
            for idx, item in enumerate(value):
                if not isinstance(item, BoundaryCondition):
                    raise TypeError(f"Item at index {idx} is not a BoundaryCondition.")
        self._set_safe("conditions", value, list)

    def add_condition(self, condition: BoundaryCondition):
        """Helper to safely append a single boundary condition."""
        current = self._conditions if self._conditions is not None else []
        if not isinstance(condition, BoundaryCondition):
            raise TypeError("Only BoundaryCondition objects can be added.")
        current.append(condition)
        self.conditions = current

@dataclass
class MaskManager(ValidatedContainer):
    __slots__ = ['_mask']
    
    # Internal state initialized to None (Zero-Debt Policy)
    _mask: np.ndarray = None

    # --- Setters and Getters with Security Firewall ---

    @property
    def mask(self) -> np.ndarray:
        return self._get_safe("mask")

    @mask.setter
    def mask(self, value: np.ndarray):
        if value is not None:
            # 1. Type and structure validation
            if not isinstance(value, np.ndarray):
                raise TypeError("Mask must be a NumPy array.")
            
            # 2. Value validation (Enforcing -1, 0, 1 enum constraint)
            if not np.all(np.isin(value, [-1, 0, 1])):
                raise ValueError("Mask must only contain values: -1 (boundary-fluid), 0 (solid), or 1 (fluid).")
        
        self._set_safe("mask", value, np.ndarray)

@dataclass
class ExternalForceManager(ValidatedContainer):
    __slots__ = ['_force_vector']
    
    # Internal state initialized to None (Zero-Debt Policy)
    _force_vector: np.ndarray = None

    # --- Setters and Getters with Security Firewall ---

    @property
    def force_vector(self) -> np.ndarray:
        return self._get_safe("force_vector")

    @force_vector.setter
    def force_vector(self, value: np.ndarray):
        if value is not None:
            # 1. Type validation
            if not isinstance(value, np.ndarray):
                raise TypeError("force_vector must be a NumPy array.")
            
            # 2. Shape validation (Min/Max items = 3)
            if value.size != 3:
                raise ValueError(f"force_vector must be 3D, got size {value.size}.")
        
        self._set_safe("force_vector", value, np.ndarray)

# =========================================================
# THE UNIVERSAL CONTAINER (The Constitution)
# =========================================================

@dataclass
class SolverState(ValidatedContainer):
    __slots__ = [
        'domain', 'grid', 'fluid', 'initial_conditions', 
        'boundary_conditions', 'external_forces', 'sim_params', 
        'masks', 'iteration', 'time', 'ready_for_time_loop'
    ]

    domain: DomainManager
    grid: GridManager
    fluid: FluidPropertiesManager
    initial_conditions: InitialConditionManager
    boundary_conditions: BoundaryConditionManager
    external_forces: ExternalForceManager
    sim_params: SimulationParameterManager
    masks: MaskManager
    
    iteration: int = 0
    time: float = 0.0
    ready_for_time_loop: bool = False