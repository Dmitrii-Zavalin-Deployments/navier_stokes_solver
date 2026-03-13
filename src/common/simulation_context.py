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
        Factory method to assemble the context from flattened data sources.
        """
        # Validate and create the input container
        input_data = SolverInput.from_dict(input_dict)
        
        # Unpack the flat dictionary directly into the SolverConfig constructor
        # This assumes config_dict matches the keys expected by SolverConfig
        config = SolverConfig(**config_dict)
        
        return cls(input_data=input_data, config=config)