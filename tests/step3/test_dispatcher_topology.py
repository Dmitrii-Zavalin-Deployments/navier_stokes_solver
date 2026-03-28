# tests/step3/test_dispatcher_topology.py

import pytest
from unittest.mock import MagicMock
from src.step3.boundaries.dispatcher import get_applicable_boundary_configs

# --- TEST FIXTURES (The 'Assets') ---

@pytest.fixture
def mock_boundary_cfg():
    """Matches your Fixed Schema Enums exactly."""
    return [
        {"location": "x_min", "type": "inflow", "values": {"u": 1.0}},
        {"location": "x_max", "type": "outflow", "values": {"p": 0.0}},
        {"location": "wall", "type": "no-slip", "values": {"u": 0.0, "v": 0.0, "w": 0.0}}
    ]

@pytest.fixture
def mock_block():
    """A StencilBlock mock with neighbors. Default: Not a ghost."""
    block = MagicMock()
    # Initialize all neighbors as NOT ghosts
    for neighbor in ['i_minus', 'i_plus', 'j_minus', 'j_plus', 'k_minus', 'k_plus']:
        getattr(block, neighbor).is_ghost = False
    # Default mask is 1 (Fluid)
    block.center.mask = 1
    return block

# --- THE TRUTH TABLE TESTS ---

def test_spatial_dispatch_xmin(mock_block, mock_boundary_cfg):
    """Scenario: Block is at the x_min boundary (detected via ghost neighbor)."""
    mock_block.i_minus.is_ghost = True
    
    result = get_applicable_boundary_configs(mock_block, mock_boundary_cfg, None, {})
    
    assert result[0]['location'] == "x_min"
    assert result[0]['type'] == "inflow"
    assert result[0]['values']['u'] == 1.0

def test_external_flow_dispatch(mock_block):
    """Scenario: EXTERNAL domain config with reference velocity."""
    mock_block.j_plus.is_ghost = True
    domain_cfg = {
        "type": "EXTERNAL",
        "reference_velocity": [10.0, 0.0, 0.5]
    }
    
    result = get_applicable_boundary_configs(mock_block, [], None, domain_cfg)
    
    assert result[0]['location'] == "y_max"
    assert result[0]['type'] == "free-stream"
    assert result[0]['values'] == {'u': 10.0, 'v': 0.0, 'w': 0.5}

def test_mask_dispatch_wall(mock_block, mock_boundary_cfg):
    """Scenario: Internal obstruction marked as -1 (Wall) in mask."""
    mock_block.center.mask = -1
    
    result = get_applicable_boundary_configs(mock_block, mock_boundary_cfg, None, {})
    
    assert result[0]['location'] == "wall"
    assert result[0]['type'] == "no-slip"

def test_mask_dispatch_solid(mock_block):
    """Scenario: Internal block marked as 0 (Solid). Hardcoded zero-velocity."""
    mock_block.center.mask = 0
    
    result = get_applicable_boundary_configs(mock_block, [], None, {})
    
    assert result[0]['location'] == "solid"
    assert result[0]['values'] == {'u': 0.0, 'v': 0.0, 'w': 0.0}

def test_interior_fallback(mock_block):
    """Scenario: Standard interior fluid cell. No ghosts, mask=1."""
    result = get_applicable_boundary_configs(mock_block, [], None, {})
    
    assert result[0]['location'] == "interior"
    assert result[0]['type'] == "fluid_gas"
    assert result[0]['values'] == {}

def test_missing_config_raises_keyerror(mock_block):
    """Scenario: x_max detected but not defined in boundary_cfg (Rule 5)."""
    mock_block.i_plus.is_ghost = True
    boundary_cfg = [{"location": "x_min", "type": "inflow"}] # missing x_max
    
    with pytest.raises(KeyError, match="Missing boundary definition for face: x_max"):
        get_applicable_boundary_configs(mock_block, boundary_cfg, None, {})

def test_external_missing_ref_velocity(mock_block):
    """Scenario: EXTERNAL type exists but reference_velocity is missing (Rule 5)."""
    mock_block.k_minus.is_ghost = True
    domain_cfg = {"type": "EXTERNAL"} # Missing required ref_v
    
    with pytest.raises(KeyError):
        get_applicable_boundary_configs(mock_block, [], None, domain_cfg)