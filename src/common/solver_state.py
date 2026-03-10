# src/solver_state.py

from dataclasses import dataclass, field
from typing import Any
import numpy as np
from src.common.base_container import ValidatedContainer

# =========================================================
# THE DEPARTMENT SAFES (Internal Management)
# =========================================================

@dataclass
class DomainManager(ValidatedContainer):
    """
    Handles Domain Configuration: INTERNAL (Pipe flow) vs EXTERNAL (Aerodynamics).
    """
    type: str = "INTERNAL"
    reference_velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))

    def __post_init__(self):
        if self.type not in ["INTERNAL", "EXTERNAL"]:
            raise ValueError(f"Invalid domain type: {self.type}. Must be INTERNAL or EXTERNAL.")

@dataclass
class GridManager(ValidatedContainer):
    """
    Step 1a: The Spatial World. 
    Encapsulates physical dimensions and resolution.
    """
    x_min: float = 0.0; x_max: float = 1.0
    y_min: float = 0.0; y_max: float = 1.0
    z_min: float = 0.0; z_max: float = 1.0
    nx: int = 10; ny: int = 10; nz: int = 10

    def __post_init__(self):
        # Validate schema minimums
        if any(n < 1 for n in [self.nx, self.ny, self.nz]):
            raise ValueError("Grid resolution (nx, ny, nz) must be at least 1.")

    # Derived spacing properties for solver usage
    @property
    def dx(self) -> float: return (self.x_max - self.x_min) / self.nx
    @property
    def dy(self) -> float: return (self.y_max - self.y_min) / self.ny
    @property
    def dz(self) -> float: return (self.z_max - self.z_min) / self.nz

@dataclass
class FluidPropertiesManager(ValidatedContainer):
    """
    Step 1c: The Physics Safe.
    Handles density (rho) and dynamic viscosity (mu).
    """
    density: float = 1.0  # rho
    viscosity: float = 0.001  # mu

    def __post_init__(self):
        if self.density <= 0:
            raise ValueError(f"Density must be greater than 0, got {self.density}")
        if self.viscosity < 0:
            raise ValueError(f"Viscosity cannot be negative, got {self.viscosity}")

@dataclass
class InitialConditionManager(ValidatedContainer):
    """
    Step 1b: The Initial Conditions Safe.
    Handles start-of-simulation velocity [u, v, w] and pressure.
    """
    velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    pressure: float = 0.0

    def __post_init__(self):
        # Ensure velocity is a 3-element array
        if self.velocity.size != 3:
            raise ValueError(f"Velocity must be a 3D array (u, v, w), got size {self.velocity.size}")

@dataclass
class SimulationParameterManager(ValidatedContainer):
    """
    Step 1e: The Chronos Guard.
    Handles simulation timing and output cadence.
    """
    time_step: float = 0.001
    total_time: float = 1.0
    output_interval: int = 100

    def __post_init__(self):
        if self.time_step <= 0:
            raise ValueError(f"time_step must be > 0, got {self.time_step}")
        if self.total_time <= 0:
            raise ValueError(f"total_time must be > 0, got {self.total_time}")
        if self.output_interval < 1:
            raise ValueError(f"output_interval must be >= 1, got {self.output_interval}")

@dataclass
class BoundaryCondition(ValidatedContainer):
    location: str # Must be in ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "wall"]
    type: str     # Must be in ["no-slip", "free-slip", "inflow", "outflow", "pressure"]
    values: dict = field(default_factory=dict) # Stores {'u', 'v', 'w', 'p'}

    def __post_init__(self):
        valid_locs = ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "wall"]
        valid_types = ["no-slip", "free-slip", "inflow", "outflow", "pressure"]
        if self.location not in valid_locs:
            raise ValueError(f"Invalid boundary location: {self.location}")
        if self.type not in valid_types:
            raise ValueError(f"Invalid boundary type: {self.type}")

@dataclass
class BoundaryConditionManager(ValidatedContainer):
    """Holds the collection of all boundary conditions."""
    conditions: list[BoundaryCondition] = field(default_factory=list)

@dataclass
class MaskManager(ValidatedContainer):
    """
    Step 1d: The Geometry Blueprint.
    Holds the canonical flattened mask of the domain.
    """
    _mask: np.ndarray = None

    @property
    def mask(self) -> np.ndarray:
        return self._mask

    @mask.setter
    def mask(self, value: np.ndarray):
        if not np.all(np.isin(value, [-1, 0, 1])):
            raise ValueError("Mask must only contain values -1, 0, or 1.")
        self._mask = value

@dataclass
class ExternalForceManager(ValidatedContainer):
    """
    Step 1f: The External Force Safe.
    Handles physical source terms like gravity or body forces.
    """
    force_vector: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))

    def __post_init__(self):
        # Enforce 3-element vector constraint (x, y, z)
        if self.force_vector.size != 3:
            raise ValueError(f"Force vector must be 3D, got size {self.force_vector.size}")

# =========================================================
# THE UNIVERSAL CONTAINER (The Constitution)
# =========================================================

@dataclass
class SolverState(ValidatedContainer):
    """
    The Project Constitution: Article 3 (The Universal State Container).
    Synchronizes input contract data with dynamic runtime buffers.
    """

    # --- 1. The Input Contract (Maps 1:1 to JSON Schema) ---
    domain: Any = None                # domain_configuration
    grid: Any = None                  # grid
    fluid: Any = None                 # fluid_properties
    boundary_conditions: Any = None   # boundary_conditions (List)
    external_forces: Any = None       # external_forces
    sim_params: Any = None            # simulation_parameters

    # --- 2. Runtime State (Hardened Buffers) ---
    fields: FieldManager = field(default_factory=FieldManager)
    topology: TopologyManager = field(default_factory=TopologyManager)
    
    # --- 3. Operators & PPE (The Engine) ---
    operators: Any = None
    advection: Any = None
    ppe: Any = None

    # --- 4. Odometer & Gate ---
    iteration: int = 0
    time: float = 0.0
    ready_for_time_loop: bool = False

    # --- 5. Flight Recorder (Diagnostics & History) ---
    health: Any = None
    history: Any = None
    diagnostics: Any = None
    manifest: Any = None

    # --- Serialization Bridge ---
    def to_json_safe(self) -> dict:
        """Contract-compliant serialization for all components."""
        return {
            "time": self.time,
            "iteration": self.iteration,
            "ready_for_time_loop": self.ready_for_time_loop,
            "domain": getattr(self.domain, "to_dict", lambda: self.domain)(),
            "grid": getattr(self.grid, "to_dict", lambda: self.grid)(),
            "fields": self.fields.to_dict(),
            "topology": self.topology.to_dict(),
            # ... add remaining components
        }