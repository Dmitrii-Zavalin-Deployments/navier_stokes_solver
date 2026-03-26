# tests/step3/test_applier_integrity.py

import logging
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.common.field_schema import FI
from src.step3.boundaries.applier import apply_boundary_values


def test_applier_enforcement_and_logging(caplog):
    """
    VERIFICATION: Ensure the Applier writes the 1e10 value to VX_STAR
    and generates the Rule 7/Rule 9 required forensic audit trail.
    """
    # 1. Setup Mock Block & Cell
    block = MagicMock()
    block.id = "ExplosionBlock_99"
    
    # Simulate the Foundation/NumPy behavior where a value might 
    # be stored/returned as a single-element array (The 'Array Leak').
    stored_values = {}
    
    def mock_set(fid, val): 
        # Store as a numpy array to test the applier's robust .item() extraction
        stored_values[fid] = np.array([val]) 
        
    def mock_get(fid): 
        return stored_values.get(fid)
    
    block.center.set_field.side_effect = mock_set
    block.center.get_field.side_effect = mock_get

    # 2. Define the 'Explosion' Rule (Using 'u' which maps to FI.VX_STAR)
    rule = {
        "location": "x_min",
        "type": "inflow",
        "values": {"u": 1e10} 
    }

    # 3. Execute with Debug Logging (Rule 7 requirement)
    with caplog.at_level(logging.DEBUG):
        apply_boundary_values(block, rule)

    # --- ASSERTIONS ---
    
    # A. Physical Memory Check
    # Verify the value was mapped to FI.VX_STAR (3)
    # We check .item() because our mock simulated the NumPy storage.
    assert stored_values[FI.VX_STAR].item() == 1e10
    
    # B. Forensic Audit Trail Check (Rule 7 Compliance)
    # Match the specific prefixes from applier.py
    assert "APPLY [Start]: Block ExplosionBlock_99" in caplog.text
    assert "Mapping u to Field 3" in caplog.text 
    
    # C. Verification & Scalar Extraction Check
    # Verify the Audit log captures the Type/Shape forensic DNA
    assert "VERIFY [Audit]: Field 3" in caplog.text
    assert "Type: <class 'numpy.ndarray'>" in caplog.text
    
    # Verify the Final Success signal with float formatting (.4e)
    assert "APPLY [Success]: Verified Field 3 = 1.0000e+10" in caplog.text


def test_applier_contract_violation(caplog):
    """Verify Rule 8: Invalid keys trigger Contract Violation error and raise KeyError."""
    block = MagicMock()
    block.id = "ErrorBlock_01"
    
    invalid_rule = {
        "location": "x_max",
        "type": "outlet",
        "values": {"invalid_key": 99.9}
    }

    # Must raise KeyError as defined in the source contract
    with caplog.at_level(logging.ERROR):
        with pytest.raises(KeyError, match="Unsupported boundary key"):
            apply_boundary_values(block, invalid_rule)
    
    # Verify the specific error message string
    assert "CONTRACT VIOLATION: Unsupported key 'invalid_key'" in caplog.text


def test_applier_missing_fields_failure(caplog):
    """Verify Strategic Failure when critical rule metadata is missing."""
    block = MagicMock()
    # Rule missing 'type'
    broken_rule = {
        "location": "y_min",
        "values": {"u": 0.0}
    }

    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValueError, match="Boundary rule missing critical fields at location="):
            apply_boundary_values(block, broken_rule)
            
    assert "STRATEGIC FAILURE" in caplog.text