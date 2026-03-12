# src/common/simulation_context.py

from dataclasses import dataclass

from src.common.solver_config import SolverConfig
from src.common.solver_input import SolverInput


@dataclass
class SimulationContext:
    """
    Acts as the primary dependency injection container for the solver.
    It encapsulates both the physical problem definition and the 
    numerical execution configuration.
    """
    input_data: SolverInput
    config: SolverConfig

    @classmethod
    def create(cls, input_dict: dict, config_dict: dict) -> "SimulationContext":
        """
        Factory method to assemble the context from separate data sources.
        """
        # Validate and create the input container
        input_data = SolverInput.from_dict(input_dict)
        
        # Assemble numerical settings
        config = SolverConfig()
        config.ppe_tolerance = config_dict["solver_settings"]["ppe_tolerance"]
        config.ppe_atol = config_dict["solver_settings"]["ppe_atol"]
        config.ppe_max_iter = config_dict["solver_settings"]["ppe_max_iter"]
        config.ppe_omega = config_dict["solver_settings"]["ppe_omega"]
        
        return cls(input_data=input_data, config=config)