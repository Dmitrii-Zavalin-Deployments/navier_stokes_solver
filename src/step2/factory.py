from .cell import Cell
from src.solver_input import SolverInput

class CellBuilder:
    def __init__(self, input_data: SolverInput):
        self.input_data = input_data

    def build(self, i: int, j: int, k: int) -> Cell:
        # Translates flat/structured input into the transient Cell object
        mask_val = self.input_data.mask.data[i, j, k]
        return Cell(x=i, y=j, z=k, mask=mask_val)
