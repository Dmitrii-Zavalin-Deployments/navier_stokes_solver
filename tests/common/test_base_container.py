# tests/common/test_base_container.py


import numpy as np
import pytest

from src.common.base_container import ValidatedContainer


# --- Setup Dummy Classes for Testing ---
class MockContainer(ValidatedContainer):
    __slots__ = ['_velocity', '_density']
    def __init__(self):
        self._velocity = None
        self._density = 1.0

class SparseMock:
    """Simulates a SciPy sparse matrix for Line 106 coverage."""
    def toarray(self):
        return np.array([[1, 0], [0, 1]])

# --- Test Cases ---

def test_get_safe_undefined_attribute_error():
    """Coverage for Line 24: AttributeError when attribute is not in slots."""
    container = MockContainer()
    with pytest.raises(AttributeError, match="not defined in MockContainer"):
        # Accessing 'pressure' which is NOT in __slots__
        container._get_safe("pressure")

def test_set_safe_type_error():
    """Coverage for Line 32: TypeError on invalid type assignment."""
    container = MockContainer()
    with pytest.raises(TypeError, match="must be <class 'float'>"):
        # Expected float, providing string
        container._set_safe("density", "heavy", float)

def test_numpy_schema_validation_error(tmp_path, mocker):
    """
    Coverage for Line 50: Explicitly trigger the NumPy diagnostic 
    by forcing jsonschema to see a raw array.
    """
    schema = {
        "type": "object",
        "properties": {"velocity": {"type": "integer"}},
        "required": ["velocity"]
    }
    schema_file = tmp_path / "schema.json"
    import json
    schema_file.write_text(json.dumps(schema))

    container = MockContainer()
    # We use a non-integer array to trigger the failure
    raw_array = np.array([1.1, 2.2])
    container._velocity = raw_array

    # MOCKING to_dict: We force to_dict to return the RAW array 
    # instead of a list, so line 48 'hasattr(failed_instance, "shape")' returns True.
    mocker.patch.object(container, 'to_dict', return_value={"velocity": raw_array})

    with pytest.raises(ValueError) as exc_info:
        container.validate_against_schema(str(schema_file))
    
    # Now this will pass because line 50 was executed
    error_msg = str(exc_info.value)
    assert "numpy.ndarray" in error_msg
    assert "shape: (2,)" in error_msg
    assert "float64" in error_msg  # Check dtype detection too

def test_to_dict_with_sparse_toarray():
    """Coverage for Line 106: Serialization of objects with .toarray()."""
    container = MockContainer()
    container._velocity = SparseMock()
    
    result = container.to_dict()
    
    # Verify it converted the sparse-mock to a nested list
    assert result["velocity"] == [[1, 0], [0, 1]]