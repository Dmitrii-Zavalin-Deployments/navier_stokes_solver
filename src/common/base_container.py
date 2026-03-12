# src/common/base_container.py
import json
from collections.abc import Iterator
from typing import Any

import jsonschema
import numpy as np


class ValidatedContainer:
    """The 'Security Guard' logic. Now with memory-efficient slots and O(1) attribute validation."""
    __slots__ = []  
    _ALLOWED_ATTRS = None

    def __iter__(self) -> Iterator[str]:
        """Helper to iterate over attributes defined in slots across the hierarchy."""
        for cls in reversed(self.__class__.__mro__):
            yield from getattr(cls, '__slots__', [])
    
    def _get_safe(self, name: str) -> Any:
        # Rule 5: Explicit or Error. No fallbacks.
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
        """Final Firewall: Validates current state against the SSoT JSON Schema."""
        with open(schema_path) as f:
            schema = json.load(f)
        jsonschema.validate(instance=self.to_dict(), schema=schema)

    def __setattr__(self, name: str, value: Any):
        if self._ALLOWED_ATTRS is None:
            allowed = set()
            for cls in self.__class__.__mro__:
                allowed.update(getattr(cls, '__slots__', []))
            self.__class__._ALLOWED_ATTRS = frozenset(allowed)
        
        # Check if the name is an allowed slot OR a property descriptor
        is_slot = name in self._ALLOWED_ATTRS
        is_property = isinstance(getattr(self.__class__, name, None), property)
        
        if not (is_slot or is_property):
            raise AttributeError(f"Memory Leak Prevention: '{name}' not in __slots__ for {self.__class__.__name__}")
        
        super().__setattr__(name, value)
    
    def to_dict(self) -> dict:
        """Serializes the container using the slots hierarchy (SSoT compliant)."""
        out = {}
        for attr in self:
            val = getattr(self, attr, None)
            if val is None:
                continue
                
            clean_key = attr.lstrip('_')
            
            # Rule 9: Hybrid Memory Foundation serialization
            if isinstance(val, ValidatedContainer):
                out[clean_key] = val.to_dict()
            elif isinstance(val, np.ndarray):
                out[clean_key] = val.tolist()
            elif hasattr(val, "toarray"):
                out[clean_key] = val.toarray().tolist()
            elif isinstance(val, dict):
                out[clean_key] = {k: (v.toarray().tolist() if hasattr(v, "toarray") 
                        else (v.tolist() if isinstance(v, np.ndarray) else v)) 
                        for k, v in val.items()}
            elif isinstance(val, list):
                out[clean_key] = [(i.to_dict() if isinstance(i, ValidatedContainer) else i) for i in val]
            else:
                out[clean_key] = val
        return out