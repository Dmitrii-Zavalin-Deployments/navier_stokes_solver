# src/common/solver_state.py

import numpy as np

from src.common.base_container import ValidatedContainer
from src.common.field_schema import FI

# =========================================================
# POST: PRE-FLIGHT INTEGRITY CHECK (Rule 9 Sentinel)
# =========================================================

def verify_foundation_integrity(state):
    """
    POST (Power-On Self-Test): Performs a 'Pre-Flight Check' on the memory wiring.
    Uses Identity Priming: Value = Index + (Field_ID / 10.0).
    Verifies that object-pointers map correctly to the monolithic fields_buffer.
    """
    if state.fields is None or state.fields.data is None:
        raise RuntimeError("POST FAILED: Fields buffer not initialized.")
        
    print("🚀 POST: Initiating Pre-Flight Memory Integrity Check...")
    
    # 1. Identity Priming & Structural Synchronicity Check
    num_cells = state.fields.data.shape[0]
    
    if len(state.stencil_matrix) != num_cells:
        raise RuntimeError(
            f"POST MISMATCH: Stencil count ({len(state.stencil_matrix)}) "
            f"!= Buffer size ({num_cells}). Architecture integrity compromised."
        )
        
    original_data = state.fields.data.copy()
    
    for f in FI:
        # Prime the buffer: Value = Index + (Field_ID / 10.0)
        state.fields.data[:, f] = np.arange(num_cells, dtype=float) + (float(f) / 10.0)

    # 2. The Inquisition: Verify pointer-to-buffer alignment
    try:
        sample_indices = [0, num_cells // 2, num_cells - 1]
        for idx in sample_indices:
            block = state.stencil_matrix[idx]
            c = block.center
            
            # Verify Pressure (FI.P = 6)
            expected_p = float(c.index) + (float(FI.P) / 10.0)
            if not np.isclose(c.p, expected_p):
                raise RuntimeError(
                    f"CRITICAL: Memory Swap! Cell {c.index} P-pointer sees {c.p}, "
                    f"expected {expected_p}. Pointer drift or mapping error detected."
                )

            # Verify Velocity X (FI.VX = 0)
            expected_vx = float(c.index) + (float(FI.VX) / 10.0)
            if not np.isclose(c.vx, expected_vx):
                raise RuntimeError(
                    f"CRITICAL: Memory Swap! Cell {c.index} VX-pointer sees {c.vx}, "
                    f"expected {expected_vx}. Pointer drift or mapping error detected."
                )

        print("✅ POST SUCCESS: Foundation wiring is verified and 'Frozen'.")

    finally:
        # 3. Restore actual simulation data
        state.fields.data[:] = original_data

# =========================================================
# THE DEPARTMENT SAFES (Memory-Hardened Managers)
# =========================================================

class DomainManager(ValidatedContainer):
    __slots__ = ['_type', '_reference_velocity']
    
    def __init__(self):
        self._type = None
        self._reference_velocity = None

    @property
    def type(self) -> str: return self._get_safe("type")
    @type.setter
    def type(self, value: str):
        if value not in ["INTERNAL", "EXTERNAL"]:
            raise ValueError(f"Invalid domain type: {value}. Must be INTERNAL or EXTERNAL.")
        self._set_safe("type", value, str)

    @property
    def reference_velocity(self) -> np.ndarray: return self._get_safe("reference_velocity")
    @reference_velocity.setter
    def reference_velocity(self, value: np.ndarray):
        if value is not None and (not isinstance(value, np.ndarray) or value.size != 3):
            raise TypeError("reference_velocity must be a 3D NumPy array.")
        self._set_safe("reference_velocity", value, np.ndarray)

class GridManager(ValidatedContainer):
    __slots__ = [
        '_x_min', '_x_max', '_y_min', '_y_max', '_z_min', '_z_max', 
        '_nx', '_ny', '_nz'
    ]

    def __init__(self):
        self._x_min = self._x_max = self._y_min = self._y_max = self._z_min = self._z_max = None
        self._nx = self._ny = self._nz = None

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

    @property
    def dx(self) -> float: return (self.x_max - self.x_min) / self.nx
    @property
    def dy(self) -> float: return (self.y_max - self.y_min) / self.ny
    @property
    def dz(self) -> float: return (self.z_max - self.z_min) / self.nz

class FluidPropertiesManager(ValidatedContainer):
    __slots__ = ['_density', '_viscosity']
    
    def __init__(self):
        self._density = self._viscosity = None

    @property
    def density(self) -> float: return self._get_safe("density")
    @density.setter
    def density(self, value: float):
        if value is not None and value <= 0: raise ValueError(f"Density must be > 0, got {value}.")
        self._set_safe("density", value, float)

    @property
    def viscosity(self) -> float: return self._get_safe("viscosity")
    @viscosity.setter
    def viscosity(self, value: float):
        if value is not None and value < 0: raise ValueError(f"Viscosity must be >= 0, got {value}.")
        self._set_safe("viscosity", value, float)

class InitialConditionManager(ValidatedContainer):
    def to_dict(self):
        return {"velocity": self.velocity.tolist(), "pressure": self.pressure}

    __slots__ = ['_velocity', '_pressure']
    
    def __init__(self):
        self._velocity = self._pressure = None

    @property
    def velocity(self) -> np.ndarray: return self._get_safe("velocity")
    @velocity.setter
    def velocity(self, value: np.ndarray):
        if value is not None and (not isinstance(value, np.ndarray) or value.size != 3):
            raise ValueError("Velocity must be a 3D NumPy array.")
        self._set_safe("velocity", value, np.ndarray)

    @property
    def pressure(self) -> float: return self._get_safe("pressure")
    @pressure.setter
    def pressure(self, value: float): self._set_safe("pressure", value, (float, int))

class SimulationParameterManager(ValidatedContainer):
    __slots__ = ['_time_step', '_total_time', '_output_interval']
    
    def __init__(self):
        self._time_step = self._total_time = self._output_interval = None

    @property
    def time_step(self) -> float: return self._get_safe("time_step")
    @time_step.setter
    def time_step(self, value: float):
        if value is not None and value <= 0: raise ValueError(f"time_step must be > 0, got {value}.")
        self._set_safe("time_step", value, (float, int))

    @property
    def total_time(self) -> float: return self._get_safe("total_time")
    @total_time.setter
    def total_time(self, value: float):
        if value is not None and value <= 0: raise ValueError(f"total_time must be > 0, got {value}.")
        self._set_safe("total_time", value, (float, int))

    @property
    def output_interval(self) -> int: return self._get_safe("output_interval")
    @output_interval.setter
    def output_interval(self, value: int):
        if value is not None and value < 1: raise ValueError(f"output_interval must be >= 1, got {value}.")
        self._set_safe("output_interval", value, int)

class BoundaryCondition(ValidatedContainer):
    __slots__ = ['_location', '_type', '_values']
    
    def __init__(self):
        self._location = self._type = self._values = None

    @property
    def location(self) -> str: return self._get_safe("location")
    @location.setter
    def location(self, value: str):
        valid = ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "wall"]
        if value is not None and value not in valid: raise ValueError(f"Invalid location '{value}'.")
        self._set_safe("location", value, str)

    @property
    def type(self) -> str: return self._get_safe("type")
    @type.setter
    def type(self, value: str):
        valid = ["no-slip", "free-slip", "inflow", "outflow", "pressure"]
        if value is not None and value not in valid: raise ValueError(f"Invalid type '{value}'.")
        self._set_safe("type", value, str)

    @property
    def values(self) -> dict: return self._get_safe("values")
    @values.setter
    def values(self, value: dict):
        if value is not None and not isinstance(value, dict): raise TypeError("values must be a dict.")
        self._set_safe("values", value, dict)

class BoundaryConditionManager(ValidatedContainer):
    def to_dict(self):
        return {"force_vector": self.force_vector.tolist()} if self.force_vector is not None else {"force_vector": [0.0, 0.0, 0.0]}

    __slots__ = ['_conditions']
    
    def __init__(self):
        self._conditions = None

    @property
    def conditions(self) -> list[BoundaryCondition]: return self._get_safe("conditions")
    @conditions.setter
    def conditions(self, value: list[BoundaryCondition]):
        if value is not None and not isinstance(value, list): raise TypeError("Must be a list.")
        self._set_safe("conditions", value, list)

    def add_condition(self, condition: BoundaryCondition):
        current = self._conditions if self._conditions is not None else []
        if not isinstance(condition, BoundaryCondition): raise TypeError("Must be BoundaryCondition.")
        current.append(condition)
        self.conditions = current

class MaskManager(ValidatedContainer):
    def to_dict(self):
        return {"force_vector": self.force_vector.tolist()} if self.force_vector is not None else {"force_vector": [0.0, 0.0, 0.0]}

    __slots__ = ['_mask']
    
    def __init__(self):
        self._mask = None

    @property
    def mask(self) -> np.ndarray: return self._get_safe("mask")
    @mask.setter
    def mask(self, value: np.ndarray):
        if value is not None:
            if not isinstance(value, np.ndarray) or not np.all(np.isin(value, [-1, 0, 1])):
                raise ValueError("Mask must be a NumPy array of -1, 0, 1.")
        self._set_safe("mask", value, np.ndarray)

class ExternalForceManager(ValidatedContainer):
    def to_dict(self):
        if self.force_vector is None:
            raise AttributeError("ExternalForceManager: force_vector must be initialized.")
        return {"force_vector": self.force_vector.tolist()}
    __slots__ = ['_force_vector']
    
    def __init__(self):
        self._force_vector = None

    @property
    def force_vector(self) -> np.ndarray: return self._get_safe("force_vector")
    @force_vector.setter
    def force_vector(self, value: np.ndarray):
        if value is not None and (not isinstance(value, np.ndarray) or value.size != 3):
            raise ValueError("force_vector must be a 3D NumPy array.")
        self._set_safe("force_vector", value, np.ndarray)

class FieldManager(ValidatedContainer):
    __slots__ = ['_data']
    
    def __init__(self):
        self._data = None

    @property
    def data(self) -> np.ndarray: return self._get_safe("data")
    @data.setter
    def data(self, value: np.ndarray):
        if not isinstance(value, np.ndarray): raise TypeError("Field data must be a NumPy array.")
        self._set_safe("data", value, np.ndarray)

    def allocate(self, n_cells: int, dtype=np.float64):
        self._data = np.zeros((n_cells, FI.num_fields()), dtype=dtype)

class ManifestManager(ValidatedContainer):
    __slots__ = ['_saved_snapshots', '_output_directory']
    
    def __init__(self):
        self._saved_snapshots = []
        self._output_directory = "output"

    @property
    def saved_snapshots(self) -> list: return self._get_safe("saved_snapshots")
    @saved_snapshots.setter
    def saved_snapshots(self, value: list): self._set_safe("saved_snapshots", value, list)

    @property
    def output_directory(self) -> str: return self._get_safe("output_directory")
    @output_directory.setter
    def output_directory(self, value: str): self._set_safe("output_directory", value, str)

# =========================================================
# THE UNIVERSAL CONTAINER (The Constitution)
# =========================================================

class SolverState(ValidatedContainer):
    __slots__ = [
        '_domain_configuration', '_grid', '_fluid_properties', '_initial_conditions', 
        '_boundary_conditions', '_external_forces', '_simulation_parameters', 
        '_mask', '_fields', '_stencil_matrix', 
        '_iteration', '_time', '_ready_for_time_loop', '_manifest'
    ]

    def __init__(self):
        super().__init__()
        self._domain_configuration = self._grid = self._fluid_properties = self._initial_conditions = None
        self._boundary_conditions = self._external_forces = self._simulation_parameters = None
        self._mask = self._fields = self._stencil_matrix = None
        self._iteration = 0
        self._time = 0.0
        self._ready_for_time_loop = False
        self.manifest = ManifestManager() 

    @property
    def manifest(self) -> ManifestManager: return self._get_safe("manifest")
    @manifest.setter
    def manifest(self, value: ManifestManager): self._set_safe("manifest", value, ManifestManager)

    @property
    def domain_configuration(self) -> DomainManager: return self._get_safe("domain_configuration")
    @domain_configuration.setter
    def domain_configuration(self, value: DomainManager): self._set_safe("domain_configuration", value, DomainManager)

    @property
    def grid(self) -> GridManager: return self._get_safe("grid")
    @grid.setter
    def grid(self, value: GridManager): self._set_safe("grid", value, GridManager)

    @property
    def fluid_properties(self) -> FluidPropertiesManager: return self._get_safe("fluid_properties")
    @fluid_properties.setter
    def fluid_properties(self, value: FluidPropertiesManager): self._set_safe("fluid_properties", value, FluidPropertiesManager)

    @property
    def initial_conditions(self) -> InitialConditionManager: return self._get_safe("initial_conditions")
    @initial_conditions.setter
    def initial_conditions(self, value: InitialConditionManager): self._set_safe("initial_conditions", value, InitialConditionManager)

    @property
    def boundary_conditions(self) -> BoundaryConditionManager: return self._get_safe("boundary_conditions")
    @boundary_conditions.setter
    def boundary_conditions(self, value: BoundaryConditionManager): self._set_safe("boundary_conditions", value, BoundaryConditionManager)

    @property
    def external_forces(self) -> ExternalForceManager: return self._get_safe("external_forces")
    @external_forces.setter
    def external_forces(self, value: ExternalForceManager): self._set_safe("external_forces", value, ExternalForceManager)

    @property
    def simulation_parameters(self) -> SimulationParameterManager: return self._get_safe("simulation_parameters")
    @simulation_parameters.setter
    def simulation_parameters(self, value: SimulationParameterManager): self._set_safe("simulation_parameters", value, SimulationParameterManager)

    @property
    def mask(self) -> MaskManager: return self._get_safe("mask")
    @mask.setter
    def mask(self, value: MaskManager): self._set_safe("mask", value, MaskManager)

    @property
    def fields(self) -> FieldManager: return self._get_safe("fields")
    @fields.setter
    def fields(self, value: FieldManager): self._set_safe("fields", value, FieldManager)

    @property
    def stencil_matrix(self) -> list: return self._get_safe("stencil_matrix")
    @stencil_matrix.setter
    def stencil_matrix(self, value: list): self._set_safe("stencil_matrix", value, list)

    @property
    def iteration(self) -> int: return self._iteration
    @iteration.setter
    def iteration(self, value: int): self._iteration = value

    @property
    def time(self) -> float: return self._time
    @time.setter
    def time(self, value: float): self._time = value

    def validate_physical_readiness(self):
        if self.fields is None or self.fields.data is None:
            raise RuntimeError("CRITICAL: Foundation buffer is missing.")
        if np.any(np.isnan(self.fields.data)) or np.any(np.isinf(self.fields.data)):
            raise RuntimeError("CRITICAL: NaNs/Infs detected in Foundation buffer!")
        if self.grid.nx is None or self.grid.nx < 1:
            raise RuntimeError("CRITICAL: Grid not properly initialized.")

    @property
    def ready_for_time_loop(self) -> bool: return self._ready_for_time_loop

    @ready_for_time_loop.setter
    def ready_for_time_loop(self, value: bool):
        if not isinstance(value, bool): raise TypeError("Must be boolean.")
        if value is True:
            verify_foundation_integrity(self)
            self.validate_physical_readiness()
        self._ready_for_time_loop = value