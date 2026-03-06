# src/common/base_container.py

import json
from typing import Any

import jsonschema
import numpy as np


class ValidatedContainer:
    """The 'Security Guard' logic. Now with runtime contract enforcement."""
    
    def _get_safe(self, name: str) -> Any:
        attr_name = f"_{name}"
        if not hasattr(self, attr_name):
            raise AttributeError(f"Coding Error: '{attr_name}' not defined in {self.__class__.__name__}.")
        val = getattr(self, attr_name)
        if val is None:
            raise RuntimeError(f"Access Error: '{name}' in {self.__class__.__name__} is uninitialized.")
        return val

    def _set_safe(self, name: str, value: Any, expected_type: type):
        if value is not None and not isinstance(value, expected_type):
            raise TypeError(f"Validation Error: '{name}' must be {expected_type}, got {type(value)}.")
        setattr(self, f"_{name}", value)

    def validate_against_schema(self, schema_path: str):
        """
        Final Firewall: Validates the current state (as a dict) 
        against the master JSON schema file.
        """
        with open(schema_path) as f:
            schema = json.load(f)
        
        # Validates current instance state against the contract
        jsonschema.validate(instance=self.to_dict(), schema=schema)

    def to_dict(self) -> dict:
        out = {}
        for attr, val in self.__dict__.items():
            clean_key = attr.lstrip('_')
            
            # 1. Handle SciPy Sparse Matrices
            if hasattr(val, "toarray"):
                out[clean_key] = val.toarray().tolist()
            
            # 2. Handle Nested Containers
            elif isinstance(val, ValidatedContainer):
                out[clean_key] = val.to_dict()
                
            # 3. Handle NumPy Arrays
            elif isinstance(val, np.ndarray):
                out[clean_key] = val.tolist()
                
            # 4. Handle Dictionaries
            elif isinstance(val, dict):
                out[clean_key] = {
                    k: (v.toarray().tolist() if hasattr(v, "toarray") 
                        else (v.tolist() if isinstance(v, np.ndarray) else v)) 
                    for k, v in val.items()
                }
            else:
                out[clean_key] = val
        return out