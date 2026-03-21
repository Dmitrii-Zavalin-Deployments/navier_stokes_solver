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
        
        config_dict.pop("dt", None)
        config = SolverConfig(**config_dict)
        
        return cls(input_data=input_data, config=config)