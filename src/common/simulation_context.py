# src/common/simulation_context.py

from dataclasses import dataclass

from src.common.solver_config import SolverConfig
from src.common.solver_input import SolverInput


@dataclass
class SimulationContext:
    """
    Acts as the primary dependency injection container for the solver.
    """
    input_data: SolverInput
    config: SolverConfig

    @classmethod
    def create(cls, input_dict: dict, config_dict: dict) -> "SimulationContext":
        """
        Factory method to assemble the context.
        """
        # 1. Load physical data
        input_data = SolverInput.from_dict(input_dict)
        
        # 2. Extract the base time_step from physical input
        # This becomes the 'Target DT' for the ElasticManager
        base_dt = input_data.simulation_parameters.time_step
        
        # 3. Inject it into the numerical config
        # Even if it's 'unchanged' elsewhere, it MUST exist here 
        # so Elasticity can compare self._dt against self.config.dt
        config_dict.pop("dt", None)
        config = SolverConfig(**config_dict)
        
        return cls(input_data=input_data, config=config)