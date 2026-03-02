# src/solver_state.py

from dataclasses import dataclass, field
from typing import Dict, Any
import numpy as np
from src.common.base_container import ValidatedContainer

# =========================================================
# STEP 1: THE DEPARTMENT SAFES
# =========================================================

@dataclass
class SolverConfig(ValidatedContainer):
    @property
    def output_directory(self):
        return "output"
    """
    Step 0: Global Instructions, Numerical Tuning, & Boundary Definitions.
    Acts as the 'Immutable Instruction Manual' for the simulation run.
    """
    case_name: str = "default_case"
    method: str = "jacobi"
    precision: str = "float64"
    
    # --- Slots for config.json / Dummy Schema data ---
    _ppe_tolerance: float = None
    _ppe_atol: float = None
    _ppe_max_iter: int = None
    
    # --- Cloned Blocks from Input Schema ---
    _simulation_parameters: dict = None  # Clones the 'simulation_parameters' block
    _boundary_conditions: list = None    # Renamed to match schema/dummies
    _external_forces: dict = None
    _initial_conditions: dict = None
    _fluid_properties: dict = None       # The storage slot for physics

    # --- PPE Property Group ---
    @property
    def ppe_tolerance(self) -> float: return self._get_safe("ppe_tolerance")
    @ppe_tolerance.setter
    def ppe_tolerance(self, v: float): self._set_safe("ppe_tolerance", v, float)

    @property
    def ppe_atol(self) -> float: return self._get_safe("ppe_atol")
    @ppe_atol.setter
    def ppe_atol(self, v: float): self._set_safe("ppe_atol", v, float)

    @property
    def ppe_max_iter(self) -> int: return self._get_safe("ppe_max_iter")
    @ppe_max_iter.setter
    def ppe_max_iter(self, v: int): self._set_safe("ppe_max_iter", v, int)

    # --- Simulation Parameters Group ---
    @property
    def simulation_parameters(self) -> dict:
        """The raw dictionary: {time_step, total_time, output_interval}"""
        return self._get_safe("simulation_parameters")
    
    @simulation_parameters.setter
    def simulation_parameters(self, v: dict):
        self._set_safe("simulation_parameters", v, dict)

    # --- Tactical Shortcuts (Facade) ---
    @property
    def dt(self) -> float: 
        return float(self.simulation_parameters["time_step"])

    @property
    def time_step(self) -> float:
        """Legacy Alias to satisfy tests expecting 'time_step' directly."""
        return self.dt

    @property
    def total_time(self) -> float:
        return float(self.simulation_parameters["total_time"])

    @property
    def output_interval(self) -> int:
        return int(self.simulation_parameters["output_interval"])

    # --- Boundary Property Group ---
    @property
    def boundary_conditions(self) -> list:
        """The 'Frozen' list of BC dictionaries from the input schema."""
        return self._get_safe("boundary_conditions")
    
    @boundary_conditions.setter
    def boundary_conditions(self, v: list):
        self._set_safe("boundary_conditions", v, list)

    # --- External Forces Group ---
    @property
    def external_forces(self) -> dict:
        """Physical source terms like gravity: {'force_vector': [x, y, z]}"""
        return self._get_safe("external_forces")

    @external_forces.setter
    def external_forces(self, v: dict):
        self._set_safe("external_forces", v, dict)

    # --- Initial Conditions Group ---
    @property
    def initial_conditions(self) -> dict:
        """The initial velocity/pressure setup: {'u': ..., 'v': ..., 'p': ...}"""
        return self._get_safe("initial_conditions")

    @initial_conditions.setter
    def initial_conditions(self, v: dict):
        self._set_safe("initial_conditions", v, dict)

    # --- Fluid Properties Group ---
    @property
    def fluid_properties(self) -> dict:
        """The raw dictionary: {'density': 1000, 'viscosity': 0.001}"""
        return self._get_safe("fluid_properties")
    
    @fluid_properties.setter
    def fluid_properties(self, v: dict):
        self._set_safe("fluid_properties", v, dict)

    # --- Tactical Shortcuts (Physical Facade) ---
    @property
    def density(self) -> float:
        """Direct access to fluid density (rho)."""
        return float(self.fluid_properties["density"])

    @property
    def viscosity(self) -> float:
        """Direct access to dynamic viscosity (mu)."""
        return float(self.fluid_properties["viscosity"])

@dataclass
class GridContext(ValidatedContainer):
    """Step 1a: The Spatial World."""
    _nx: int = None; _ny: int = None; _nz: int = None
    _x_min: float = None; _x_max: float = None
    _y_min: float = None; _y_max: float = None
    _z_min: float = None; _z_max: float = None

    @property
    def nx(self) -> int: return self._get_safe("nx")
    @nx.setter
    def nx(self, val: int): self._set_safe("nx", val, int)

    @property
    def ny(self) -> int: return self._get_safe("ny")
    @ny.setter
    def ny(self, val: int): self._set_safe("ny", val, int)

    @property
    def nz(self) -> int: return self._get_safe("nz")
    @nz.setter
    def nz(self, val: int): self._set_safe("nz", val, int)

    @property
    def x_min(self) -> float: return self._get_safe("x_min")
    @x_min.setter
    def x_min(self, val: float): self._set_safe("x_min", val, float)

    @property
    def x_max(self) -> float: return self._get_safe("x_max")
    @x_max.setter
    def x_max(self, val: float): self._set_safe("x_max", val, float)

    @property
    def y_min(self) -> float: return self._get_safe("y_min")
    @y_min.setter
    def y_min(self, val: float): self._set_safe("y_min", val, float)

    @property
    def y_max(self) -> float: return self._get_safe("y_max")
    @y_max.setter
    def y_max(self, val: float): self._set_safe("y_max", val, float)

    @property
    def z_min(self) -> float: return self._get_safe("z_min")
    @z_min.setter
    def z_min(self, val: float): self._set_safe("z_min", val, float)

    @property
    def z_max(self) -> float: return self._get_safe("z_max")
    @z_max.setter
    def z_max(self, val: float): self._set_safe("z_max", val, float)

    # Derived spacing properties
    @property
    def dx(self) -> float: 
        return float((self.x_max - self.x_min) / self.nx)

    @property
    def dy(self) -> float: 
        return float((self.y_max - self.y_min) / self.ny)

    @property
    def dz(self) -> float: 
        return float((self.z_max - self.z_min) / self.nz)

    # Performance Shortcuts: Inverse Spacing
    @property
    def inv_dx(self) -> float:
        return 1.0 / self.dx

    @property
    def inv_dy(self) -> float:
        return 1.0 / self.dy

    @property
    def inv_dz(self) -> float:
        return 1.0 / self.dz

@dataclass
class FieldData(ValidatedContainer):
    """
    Step 1, 3 & 4: The Memory Map. 
    Contains both the core physical fields and the extended boundary workhorses.

    * **state.fields.U, V, W**: The corrected, divergence-free velocity vectors.
    * **state.fields.P**: The final pressure field.
    * **state.fields.U_star, V_star, W_star**: Predictor fields (Step 3 intermediate).
    * **state.fields.U_ext, V_ext, W_ext, P_ext**: Extended arrays (with ghost cells).
    """
    # --- Core Fields (Step 1 & 3) ---
    _P: np.ndarray = None; _U: np.ndarray = None
    _V: np.ndarray = None; _W: np.ndarray = None

    # --- Predictor Fields (Step 3 Intermediate) ---
    _U_star: np.ndarray = None; _V_star: np.ndarray = None; _W_star: np.ndarray = None

    # --- Extended Fields (Step 4) ---
    _P_ext: np.ndarray = None; _U_ext: np.ndarray = None
    _V_ext: np.ndarray = None; _W_ext: np.ndarray = None

    # --- Core Properties ---
    @property
    def P(self) -> np.ndarray: return self._get_safe("P")
    @P.setter
    def P(self, val: np.ndarray): self._set_safe("P", val, np.ndarray)

    @property
    def U(self) -> np.ndarray: return self._get_safe("U")
    @U.setter
    def U(self, val: np.ndarray): self._set_safe("U", val, np.ndarray)

    @property
    def V(self) -> np.ndarray: return self._get_safe("V")
    @V.setter
    def V(self, val: np.ndarray): self._set_safe("V", val, np.ndarray)

    @property
    def W(self) -> np.ndarray: return self._get_safe("W")
    @W.setter
    def W(self, val: np.ndarray): self._set_safe("W", val, np.ndarray)

    # --- Predictor Properties ---
    @property
    def U_star(self) -> np.ndarray: return self._get_safe("U_star")
    @U_star.setter
    def U_star(self, v: np.ndarray): self._set_safe("U_star", v, np.ndarray)

    @property
    def V_star(self) -> np.ndarray: return self._get_safe("V_star")
    @V_star.setter
    def V_star(self, v: np.ndarray): self._set_safe("V_star", v, np.ndarray)

    @property
    def W_star(self) -> np.ndarray: return self._get_safe("W_star")
    @W_star.setter
    def W_star(self, v: np.ndarray): self._set_safe("W_star", v, np.ndarray)

    # --- Extended Properties ---
    @property
    def P_ext(self) -> np.ndarray: return self._get_safe("P_ext")
    @P_ext.setter
    def P_ext(self, val: np.ndarray): self._set_safe("P_ext", val, np.ndarray)

    @property
    def U_ext(self) -> np.ndarray: return self._get_safe("U_ext")
    @U_ext.setter
    def U_ext(self, val: np.ndarray): self._set_safe("U_ext", val, np.ndarray)

    @property
    def V_ext(self) -> np.ndarray: return self._get_safe("V_ext")
    @V_ext.setter
    def V_ext(self, val: np.ndarray): self._set_safe("V_ext", val, np.ndarray)

    @property
    def W_ext(self) -> np.ndarray: return self._get_safe("W_ext")
    @W_ext.setter
    def W_ext(self, val: np.ndarray): self._set_safe("W_ext", val, np.ndarray)

    def _get_mapping(self) -> dict:
        """Internal helper for dictionary-style key translation."""
        return {
            "U": "U", "V": "V", "W": "W", "P": "P",
            "U_STAR": "U_star", "V_STAR": "V_star", "W_STAR": "W_star",
            "U_EXT": "U_ext", "V_EXT": "V_ext", "W_EXT": "W_ext", "P_EXT": "P_ext"
        }

    def __getitem__(self, key: str) -> np.ndarray:
        """
        Step 3 Support: Allows reading fields using dictionary syntax.
        Example: velocity = state.fields["U_star"]
        """
        target = self._get_mapping().get(key.upper())
        if target:
            return getattr(self, target)
        raise KeyError(f"Field '{key}' is not recognized.")

    def __setitem__(self, key: str, value: np.ndarray):
        """
        Step 3 & 4 Support: Handles both core, predictor, and extended assignments.
        Example: state.fields["U_star"] = intermediate_u
        """
        target = self._get_mapping().get(key.upper())
        if target:
            setattr(self, target, value)
        else:
            raise KeyError(f"Field '{key}' is not recognized.")

@dataclass
class MaskData(ValidatedContainer):
    """
    Step 1d: The Geometry Blueprint.
    Refined specifically from map_geometry_mask outputs.
    """
    _mask: np.ndarray = None            
    _is_fluid: np.ndarray = None        
    _is_boundary: np.ndarray = None     

    @property
    def mask(self) -> np.ndarray: return self._get_safe("mask")
    @mask.setter
    def mask(self, v: np.ndarray): self._set_safe("mask", v, np.ndarray)

    @property
    def is_fluid(self) -> np.ndarray: return self._get_safe("is_fluid")
    @is_fluid.setter
    def is_fluid(self, v: np.ndarray): self._set_safe("is_fluid", v, np.ndarray)

    @property
    def is_boundary(self) -> np.ndarray: return self._get_safe("is_boundary")
    @is_boundary.setter
    def is_boundary(self, v: np.ndarray): self._set_safe("is_boundary", v, np.ndarray)

@dataclass
class FluidProperties(ValidatedContainer):
    """Step 1c: The Physics Safe."""
    _rho: float = None; _mu: float = None

    @property
    def rho(self) -> float: return self._get_safe("rho")
    @rho.setter
    def rho(self, val: float): self._set_safe("rho", val, float)

    @property
    def mu(self) -> float: return self._get_safe("mu")
    @mu.setter
    def mu(self, val: float): self._set_safe("mu", val, float)

@dataclass
class OperatorStorage(ValidatedContainer):
    """Step 2c: The Sparse Matrix Engine (Calculus Operators)."""
    _divergence: Any = None
    _grad_x: Any = None; _grad_y: Any = None; _grad_z: Any = None
    _laplacian: Any = None

    @property
    def divergence(self) -> Any: return self._get_safe("divergence")
    @divergence.setter
    def divergence(self, v: Any): setattr(self, "_divergence", v)

    @property
    def grad_x(self) -> Any: return self._get_safe("grad_x")
    @grad_x.setter
    def grad_x(self, v: Any): setattr(self, "_grad_x", v)

    @property
    def grad_y(self) -> Any: return self._get_safe("grad_y")
    @grad_y.setter
    def grad_y(self, v: Any): setattr(self, "_grad_y", v)

    @property
    def grad_z(self) -> Any: return self._get_safe("grad_z")
    @grad_z.setter
    def grad_z(self, v: Any): setattr(self, "_grad_z", v)

    @property
    def laplacian(self) -> Any: return self._get_safe("laplacian")
    @laplacian.setter
    def laplacian(self, v: Any): setattr(self, "_laplacian", v)

@dataclass
class AdvectionStructure(ValidatedContainer):
    """Step 2b: Momentum Transport Stencils."""
    _weights: Any = None
    _indices: Any = None

    @property
    def weights(self) -> Any: return self._get_safe("weights")
    @weights.setter
    def weights(self, v: Any): self._set_safe("weights", v, (np.ndarray, object))

    @property
    def indices(self) -> Any: return self._get_safe("indices")
    @indices.setter
    def indices(self, v: Any): self._set_safe("indices", v, (np.ndarray, object))

@dataclass
class PPEContext(ValidatedContainer):
    """
    Step 2d & 3: Pressure Poisson Equation System.
    These remain 'Hot' in memory to avoid costly re-assembly during Step 3.
    """
    _A: Any = None
    _preconditioner: Any = None

    @property
    def A(self) -> Any: 
        """
        The system matrix (LHS). Usually a large, sparse Laplacian-based 
        matrix representing the linear equations for pressure.
        """
        return self._get_safe("A")
    @A.setter
    def A(self, v: Any): self._set_safe("A", v, object)

    @property
    def preconditioner(self) -> Any: 
        """
        The mathematical 'shortcut' (e.g., Jacobi, ILU) used to 
        accelerate the convergence of the Pressure solver.
        """
        return self._get_safe("preconditioner")
    @preconditioner.setter
    def preconditioner(self, v: Any): self._set_safe("preconditioner", v, object)

@dataclass
class SimulationHealth(ValidatedContainer):
    """
    Step 2e & 3: Current "Vitals".
    Describes the state of the simulation at this exact moment.

    ## 4. Post-Flight Health Check (`state.health`)

    Generated by `finalize_simulation_health(state)`, this provides a high-level 
    summary of whether the run was a success or a failure.

    * **state.health.is_stable**: A final bool. If a NaN occurred at step 500 of 10,000, this will be False.
    * **state.health.max_u**: The absolute maximum velocity recorded across the entire volume at the final step.
    * **state.health.post_correction_divergence_norm**: The final proof of mass conservation (should be approx 0).
    """
    _max_u: float = None
    _divergence_norm: float = None
    _is_stable: bool = None
    _post_correction_divergence_norm: float = None

    @property
    def max_u(self) -> float: 
        """The speed of the fastest point in the simulation (exploding check)."""
        return self._get_safe("max_u")
    @max_u.setter
    def max_u(self, v: float): self._set_safe("max_u", v, float)

    @property
    def divergence_norm(self) -> float: 
        """Measures how 'un-physical' the initial or current flow field is."""
        return self._get_safe("divergence_norm")
    @divergence_norm.setter
    def divergence_norm(self, v: float): self._set_safe("divergence_norm", v, float)

    @property
    def is_stable(self) -> bool: 
        """Boolean flag: flips to False if NaNs or Infs are detected."""
        return self._get_safe("is_stable")
    @is_stable.setter
    def is_stable(self, v: bool): self._set_safe("is_stable", v, bool)

    @property
    def post_correction_divergence_norm(self) -> float:
        """Proves the pressure correction worked and mass is conserved (Target ~0)."""
        return self._get_safe("post_correction_divergence_norm")
    @post_correction_divergence_norm.setter
    def post_correction_divergence_norm(self, v: float): 
        self._set_safe("post_correction_divergence_norm", v, float)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Step 3 Support: Compatibility shim for dictionary-style access.
        Allows: state.health.get("max_velocity_magnitude")
        """
        mapping = {
            "post_correction_divergence_norm": "_post_correction_divergence_norm",
            "max_velocity_magnitude": "_max_u",
            "divergence_norm": "_divergence_norm",
            "is_stable": "_is_stable"
        }
        attr = mapping.get(key)
        if attr:
            # We use getattr directly to avoid the 'None' check of _get_safe 
            # if the user provided a default.
            val = getattr(self, attr, None)
            return val if val is not None else default
        return default

@dataclass
class SimulationHistory(ValidatedContainer):
    """
    Step 3 & 5: The Black Box Flight Recorder.
    Tracks the evolution of the simulation over time for plotting and 
    post-processing.
    """
    _times: list = field(default_factory=list)
    _divergence_norms: list = field(default_factory=list)
    _max_velocity_history: list = field(default_factory=list)
    _ppe_status_history: list = field(default_factory=list)
    _energy_history: list = field(default_factory=list)

    @property
    def times(self) -> list: 
        """A list of every time-stamp simulated so far."""
        return self._get_safe("times")

    @property
    def divergence_norms(self) -> list: 
        """A record of how the mass conservation error changed over time."""
        return self._get_safe("divergence_norms")

    @property
    def max_velocity_history(self) -> list: 
        """A graph-ready list of peak speeds (Stability tracking)."""
        return self._get_safe("max_velocity_history")

    @property
    def ppe_status_history(self) -> list: 
        """A list of solver outcomes (e.g., 'Converged', 'Failed')."""
        return self._get_safe("ppe_status_history")

    @property
    def energy_history(self) -> list: 
        """The total kinetic energy in the system (Physical decay/growth check)."""
        return self._get_safe("energy_history")

    def __getitem__(self, key: str) -> list:
        """Allows dictionary-style access: history['times']."""
        mapping = {
            "times": self.times,
            "divergence_norms": self.divergence_norms,
            "max_velocity_history": self.max_velocity_history,
            "ppe_status_history": self.ppe_status_history,
            "energy_history": self.energy_history
        }
        if key in mapping:
            return mapping[key]
        raise KeyError(f"History key '{key}' not found.")

@dataclass
class Diagnostics(ValidatedContainer):
    """
    Step 4: Pre-Flight Readiness Audit.
    Summarizes memory usage and BC integrity before the heavy lifting begins.
    """
    _memory_footprint_gb: float = None
    _bc_verification_passed: bool = None
    _initial_cfl_dt: float = None
    _source_term_applied: bool = True
 
    @property
    def source_term_applied(self) -> bool: return self._get_safe("source_term_applied")

    @property
    def memory_footprint_gb(self) -> float: 
        """
        Total RAM usage of the extended arrays in GB.
        Calculated as: (Total Voxels * 8 bytes per float64 * Num Fields) / 1e9.
        """
        return self._get_safe("memory_footprint_gb")
    @memory_footprint_gb.setter
    def memory_footprint_gb(self, v: float): self._set_safe("memory_footprint_gb", v, float)

    @property
    def bc_verification_passed(self) -> bool: 
        """Confirms ghost cells were correctly populated by Step 4."""
        return self._get_safe("bc_verification_passed")
    @bc_verification_passed.setter
    def bc_verification_passed(self, v: bool): self._set_safe("bc_verification_passed", v, bool)

    @property
    def initial_cfl_dt(self) -> float: 
        """Maximum allowable time-step based on initial conditions."""
        return self._get_safe("initial_cfl_dt")
    @initial_cfl_dt.setter
    def initial_cfl_dt(self, v: float): self._set_safe("initial_cfl_dt", v, float)

    def __getitem__(self, key: str) -> Any:
        """Translates legacy names and uses the 'Security Guard' to fetch them."""
        mapping = {
            "memory_footprint": "memory_footprint_gb",
            "bc_verification": "bc_verification_passed",
            "cfl_limit": "initial_cfl_dt",
            "source_term_applied": "source_term_applied"
        }
        # If the key is a legacy alias, use the new name; else use the key as-is.
        target = mapping.get(key, key)
        
        # Route through the actual Security Guard logic
        return self._get_safe(target)

    def get(self, key: str, default: Any = None) -> Any:
        """A safe wrapper around __getitem__ that respects the default value."""
        try:
            return self[key]
        except (AttributeError, RuntimeError, KeyError):
            return default

@dataclass
class OutputManifest(ValidatedContainer):
    """
    ## 5. Output Artifacts (Manifest)
    
    This tracks the physical location of files generated during Step 5.
    Essential for archiving logic and post-processing verification.
    """
    _output_directory: str = "output"
    _saved_snapshots: list[str] = field(default_factory=list)
    _final_checkpoint: str = None
    _log_file: str = None

    @property
    def output_directory(self) -> str:
        """The base folder where all results are stored."""
        return self._get_safe("output_directory")
    
    @output_directory.setter
    def output_directory(self, v: str): 
        self._set_safe("output_directory", v, str)

    @property
    def saved_snapshots(self) -> list[str]:
        """A list of full paths to every VTK/HDF5 file created."""
        return self._get_safe("saved_snapshots")

    @saved_snapshots.setter
    def saved_snapshots(self, v: list):
        """Allows bulk update of the snapshot registry (required by Step 5 assembly)."""
        self._set_safe("saved_snapshots", v, list)

    @property
    def final_checkpoint(self) -> str:
        """The path to the last .npy or .h5 state for restarting."""
        return self._get_safe("final_checkpoint")
    
    @final_checkpoint.setter
    def final_checkpoint(self, v: str): 
        self._set_safe("final_checkpoint", v, str)

    @property
    def log_file(self) -> str:
        """Path to the .txt or .log file containing solver convergence stats."""
        return self._get_safe("log_file")
    
    @log_file.setter
    def log_file(self, v: str): 
        self._set_safe("log_file", v, str)

# =========================================================
# THE UNIVERSAL CONTAINER (The Constitution)
# =========================================================

@dataclass
class SolverState:
    @property
    def constants(self):
        return self.fluid
    """
    The Project Constitution: Article 3 (The Universal State Container).
    
    This central container synchronizes all data across the five-step pipeline:
    1. Domain Setup -> 2. Operator Assembly -> 3. Iterative Loop -> 
    4. Boundary Enforcement -> 5. Artifact Export.
    """

    # --- 1. Hardened Safe Objects ---
    config: SolverConfig = field(default_factory=SolverConfig)
    grid: GridContext = field(default_factory=GridContext)
    fields: FieldData = field(default_factory=FieldData)
    masks: MaskData = field(default_factory=MaskData)
    fluid: FluidProperties = field(default_factory=FluidProperties)
    operators: OperatorStorage = field(default_factory=OperatorStorage)
    advection: AdvectionStructure = field(default_factory=AdvectionStructure)
    ppe: PPEContext = field(default_factory=PPEContext)
    health: SimulationHealth = field(default_factory=SimulationHealth)
    history: SimulationHistory = field(default_factory=SimulationHistory)
    diagnostics: Diagnostics = field(default_factory=Diagnostics)
    manifest: OutputManifest = field(default_factory=OutputManifest)

    # --- 2. Global Simulation Odometer ---
    iteration: int = 0
    time: float = 0.0

    # --- 3. Execution Gate ---
    ready_for_time_loop: bool = False

    # ---------------------------------------------------------
    # Attribute Interface (Facade)
    # ---------------------------------------------------------
    
    @property
    def pressure(self) -> np.ndarray:
        return self.fields.P

    @property
    def velocity_u(self) -> np.ndarray:
        return self.fields.U

    @property
    def velocity_v(self) -> np.ndarray:
        return self.fields.V

    @property
    def velocity_w(self) -> np.ndarray:
        return self.fields.W

    @property
    def is_fluid(self) -> np.ndarray:
        return self.masks.is_fluid

    # --- Physics Delegation ---
    @property
    def rho(self) -> float:
        """Delegates to config.density."""
        return self.config.density

    @property
    def mu(self) -> float:
        """Delegates to config.viscosity."""
        return self.config.viscosity

    # --- Step 4 Extended Field Shortcuts ---
    @property
    def U_ext(self) -> np.ndarray: return self.fields.U_ext
    @property
    def V_ext(self) -> np.ndarray: return self.fields.V_ext
    @property
    def W_ext(self) -> np.ndarray: return self.fields.W_ext
    @property
    def P_ext(self) -> np.ndarray: return self.fields.P_ext

    # --- Numerical Shortcuts (Validated by Config & Grid) ---
    @property
    def dt(self) -> float:
        return self.config.dt

    @property
    def inv_dx(self) -> float:
        return self.grid.inv_dx

    @property
    def inv_dy(self) -> float:
        return self.grid.inv_dy

    @property
    def inv_dz(self) -> float:
        return self.grid.inv_dz

    # ---------------------------------------------------------
    # Legacy & Pipeline Support
    # ---------------------------------------------------------
    @property
    def step4_diagnostics(self) -> Diagnostics:
        return self.diagnostics

    def to_legacy_dict(self) -> Dict[str, Any]:
        return {
            "U_ext": self.U_ext,
            "V_ext": self.V_ext,
            "W_ext": self.W_ext,
            "P_ext": self.P_ext,
            "step4_diagnostics": self.step4_diagnostics,
            "ready_for_time_loop": self.ready_for_time_loop
        }

    # ---------------------------------------------------------
    # Serialization (Contract Bridge)
    # ---------------------------------------------------------
    def to_json_safe(self) -> dict:
        return {
            "time": self.time,
            "iteration": self.iteration,
            "ready_for_time_loop": self.ready_for_time_loop,
            "config": self.config.to_dict(),
            "grid": self.grid.to_dict(),
            "fields": self.fields.to_dict(),
            "masks": self.masks.to_dict(),
            "fluid": self.fluid.to_dict(),
            "operators": self.operators.to_dict(),
            "advection": self.advection.to_dict(),
            "ppe": self.ppe.to_dict(),
            "health": self.health.to_dict(),
            "history": self.history.to_dict(),
            "diagnostics": self.diagnostics.to_dict(),
            "manifest": self.manifest.to_dict()
        }