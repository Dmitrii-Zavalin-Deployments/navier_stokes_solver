# tests/step3/test_applier_integrity.py

import logging
from unittest.mock import MagicMock

import pytest

from src.common.field_schema import FI
from src.step3.boundaries.applier import apply_boundary_values


def test_applier_enforcement_and_logging(caplog):
    """
    VERIFICATION: Ensure the Applier writes the 1e10 value to the correct FI.VX_STAR buffer
    and generates the required audit trail.
    """
    # 1. Setup Mock Block & Cell
    block = MagicMock()
    block.id = "ExplosionBlock_99"
    
    # We need to simulate the set/get behavior of the cell memory
    stored_values = {}
    def mock_set(fid, val): stored_values[fid] = val
    def mock_get(fid): return stored_values.get(fid)
    
    block.center.set_field.side_effect = mock_set
    block.center.get_field.side_effect = mock_get

    # 2. Define the 'Explosion' Rule (1e10)
    rule = {
        "location": "x_min",
        "type": "inflow",
        "values": {"u": 1e10} # Velocity in x-direction
    }

    # 3. Execute with Debug Logging
    with caplog.at_level(logging.DEBUG):
        apply_boundary_values(block, rule)

    # --- ASSERTIONS ---
    
    # A. Physical Memory Check
    # Did the value actually land in FI.VX_STAR (ID 3)?
    assert stored_values[FI.VX_STAR] == 1e10
    
    # B. Logger Evidence Check (Decision Trace)
    assert "APPLY [Start]: Block ExplosionBlock_99" in caplog.text
    assert "Mapping u to Field 3" in caplog.text # FI.VX_STAR is typically 3
    assert "Value 1.0000e+10" in caplog.text
    
    # C. Verification Log Check (The 'Success' signal)
    assert "APPLY [Success]: Verified Field 3 = 1.0000e+10" in caplog.text

def test_applier_contract_violation(caplog):
    """Verify that invalid keys trigger the error logger and raise KeyError."""
    block = MagicMock()
    invalid_rule = {
        "location": "x_max",
        "type": "outlet",
        "values": {"invalid_key": 99.9}
    }

    with caplog.at_level(logging.ERROR):
        with pytest.raises(KeyError):
            apply_boundary_values(block, invalid_rule)
    
    assert "CONTRACT VIOLATION: Unsupported key 'invalid_key'" in caplog.text