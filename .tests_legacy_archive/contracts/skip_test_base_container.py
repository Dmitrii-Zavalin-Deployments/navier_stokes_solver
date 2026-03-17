# tests/contracts/test_base_container.py

import numpy as np
import pytest

from src.common.base_container import ValidatedContainer


class MockContainer(ValidatedContainer):
    """A minimal implementation to stress-test the Security Guard."""
    def __init__(self):
        self._velocity = None  # Defined but uninitialized
        self._metadata = {"key": "value"}

def test_base_container_attribute_error():
    """Hits Line 13: AttributeError for non-existent internal slots."""
    container = MockContainer()
    with pytest.raises(AttributeError, match="not defined"):
        # Attempting to access 'pressure' when '_pressure' doesn't exist
        container._get_safe("pressure")

def test_base_container_type_error():
    """Hits Line 18: TypeError for incorrect type assignment."""
    container = MockContainer()
    # If we expected a float but got a string
    with pytest.raises(TypeError, match="Validation Error"):
        container._set_safe("velocity", "not_a_float", float)

def test_base_container_to_dict_recursion():
    """Hits Line 26: dict comprehension logic for nested structures."""
    container = MockContainer()
    # Add a numpy array inside a dictionary to trigger the complex path in to_dict
    container._metadata = {
        "array_data": np.array([1, 2]),
        "simple_data": 42
    }
    
    result = container.to_dict()
    assert isinstance(result["metadata"]["array_data"], list)
    assert result["metadata"]["simple_data"] == 42

def test_base_container_numpy_conversion():
    """Ensures top-level numpy arrays are converted to lists."""
    container = MockContainer()
    container._set_safe("velocity", np.array([1.0, 2.0]), np.ndarray)
    
    result = container.to_dict()
    assert result["velocity"] == [1.0, 2.0]