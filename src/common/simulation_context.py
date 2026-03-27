# src/common/simulation_context.py

from dataclasses import dataclass

from src.common.solver_config import SolverConfig
from src.common.solver_input import SolverInput


@dataclass(slots=True)  # Rule 0: Mandatory __slots__ for memory efficiency
class SimulationContext:
    """
    Acts as the primary dependency injection container for the solver.
    Compliance: Rule 4 (SSoT) & Rule 0 (Performance).
    """
    input_data: SolverInput
    config: SolverConfig

    @classmethod
    def create(cls, input_dict: dict, config_dict: dict) -> "SimulationContext":
        # Rule 4: Extract physical data first
        input_data = SolverInput.from_dict(input_dict)

        # Rule 5: Ensure 'dt' doesn't contaminate the static config object
        # This prevents "Hidden Defaults" or redundant state drift.
        config_dict.pop("dt", None)
        config = SolverConfig(**config_dict)

        return cls(input_data=input_data, config=config)