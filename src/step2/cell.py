from dataclasses import dataclass

@dataclass
class Cell:
    x: int
    y: int
    z: int
    mask: int = 0
    # Add other properties as needed for validation