# src/common/solver_state.py

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.common.base_container import ValidatedContainer

# =========================================================
# THE DEPARTMENT SAFES (Input Validation & Management)
# =========================================================

@dataclass
class DomainManager(ValidatedContainer):
    """Handles Domain Configuration: INTERNAL vs EXTERNAL."""
    type: str = "INTERNAL"
    reference_velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))

    def __post_init__(self):
        if self.type not in ["INTERNAL", "EXTERNAL"]:
            raise ValueError(f"Invalid domain type: {self.type}. Must be INTERNAL or EXTERNAL.")

@dataclass
class GridManager(ValidatedContainer):
    """Encapsulates spatial dimensions and resolution."""
    x_min: float = 0.0; x_max: float = 1.0
    y_min: float = 0.0; y_max: float = 1.0
    z_min: float = 0.0; z_max: float = 1.0
    nx: int = 10; ny: int = 10; nz: int = 10

    def __post_init__(self):
        if any(n < 1 for n in [self.nx, self.ny, self.nz]):
            raise ValueError("Grid resolution (nx, ny, nz) must be at least 1.")

    @property
    def dx(self) -> float: return (self.x_max - self.x_min) / self.nx
    @property
    def dy(self) -> float: return (self.y_max - self.y_min) / self.ny
    @property
    def dz(self) -> float: return (self.z_max - self.z_min) / self.nz

@dataclass
class FluidPropertiesManager(ValidatedContainer):
    """Handles physical properties: density and viscosity."""
    density: float = 1.0
    viscosity: float = 0.001

    def __post_init__(self):
        if self.density <= 0:
            raise ValueError(f"Density must be > 0, got {self.density}")
        if self.viscosity < 0:
            raise ValueError(f"Viscosity must be >= 0, got {self.viscosity}")

@dataclass
class InitialConditionManager(ValidatedContainer):
    """Handles simulation start state."""
    velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    pressure: float = 0.0

    def __post_init__(self):
        if self.velocity.size != 3:
            raise ValueError(f"Velocity must be 3D, got size {self.velocity.size}")

@dataclass
class SimulationParameterManager(ValidatedContainer):
    """Handles simulation timing and output cadence."""
    time_step: float = 0.001
    total_time: float = 1.0
    output_interval: int = 100

    def __post_init__(self):
        if self.time_step <= 0 or self.total_time <= 0:
            raise ValueError("time_step and total_time must be > 0.")
        if self.output_interval < 1:
            raise ValueError("output_interval must be >= 1.")

@dataclass
class BoundaryCondition(ValidatedContainer):
    location: str
    type: str
    values: dict = field(default_factory=dict)

    def __post_init__(self):
        valid_locs = ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "wall"]
        valid_types = ["no-slip", "free-slip", "inflow", "outflow", "pressure"]
        if self.location not in valid_locs:
            raise ValueError(f"Invalid location: {self.location}")
        if self.type not in valid_types:
            raise ValueError(f"Invalid type: {self.type}")

@dataclass
class BoundaryConditionManager(ValidatedContainer):
    conditions: list[BoundaryCondition] = field(default_factory=list)

@dataclass
class MaskManager(ValidatedContainer):
    """Holds the canonical flattened geometry mask."""
    _mask: np.ndarray = None

    @property
    def mask(self) -> np.ndarray: return self._mask

    @mask.setter
    def mask(self, value: np.ndarray):
        if not np.all(np.isin(value, [-1, 0, 1])):
            raise ValueError("Mask must contain only -1, 0, or 1.")
        self._mask = value

@dataclass
class ExternalForceManager(ValidatedContainer):
    """Handles body forces."""
    force_vector: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))

    def __post_init__(self):
        if self.force_vector.size != 3:
            raise ValueError("Force vector must be 3D.")

# =========================================================
# THE UNIVERSAL CONTAINER (The Constitution)
# =========================================================

@dataclass
class SolverState(ValidatedContainer):
    """The central registry for all validated simulation data."""

    # Input Contract
    domain: DomainManager = field(default_factory=DomainManager)
    grid: GridManager = field(default_factory=GridManager)
    fluid: FluidPropertiesManager = field(default_factory=FluidPropertiesManager)
    initial_conditions: InitialConditionManager = field(default_factory=InitialConditionManager)
    boundary_conditions: BoundaryConditionManager = field(default_factory=BoundaryConditionManager)
    external_forces: ExternalForceManager = field(default_factory=ExternalForceManager)
    sim_params: SimulationParameterManager = field(default_factory=SimulationParameterManager)

    # Runtime State
    masks: MaskManager = field(default_factory=MaskManager)
    
    # Engine & Odometer
    iteration: int = 0
    time: float = 0.0
    ready_for_time_loop: bool = False

    # Serialization Bridge
    def to_json_safe(self) -> dict:
        """Returns a contract-compliant dictionary representation."""
        return {
            "time": self.time,
            "iteration": self.iteration,
            "domain": self.domain.to_dict(),
            "grid": self.grid.to_dict(),
            "fluid": self.fluid.to_dict(),
            "initial_conditions": self.initial_conditions.to_dict(),
            "boundary_conditions": self.boundary_conditions.to_dict(),
            "external_forces": self.external_forces.to_dict(),
            "sim_params": self.sim_params.to_dict(),
        }