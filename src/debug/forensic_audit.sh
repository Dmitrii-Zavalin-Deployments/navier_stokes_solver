# =========================
# 0. Context: failing tests
# =========================
echo "=== Failing tests (reconfirm) ==="
pytest -q tests/quality_gates/logic_gate/test_step3_mms.py::test_logic_gate_3_center_mutation_audit \
          tests/quality_gates/sensitivity_gate/test_bc_collisions.py::test_gate_3a_3b_dispatcher_mask_symmetry \
          tests/quality_gates/sensitivity_gate/test_bc_collisions.py::test_gate_3c_interior_fluid_axiom \
          tests/quality_gates/sensitivity_gate/test_bc_collisions.py::test_gate_3a_missing_wall_config_collision -q

# ==========================================
# 1. See how the dummy state & masks are built
# ==========================================
echo
echo "=== make_step2_output_dummy wiring ==="
grep -RIn "make_step2_output_dummy" -n tests src || true
cat -n tests/helpers/solver_step2_output_dummy.py

echo
echo "=== Step 1 mask + padded mask wiring ==="
cat -n src/step1/helpers.py
cat -n src/step1/orchestrate_step1.py | sed -n '70,110p'

echo
echo "=== Step 2 stencil assembly (where ghosts & centers are wired) ==="
cat -n src/step2/stencil_assembler.py

# =========================================================
# 2. Inspect the exact blocks used in the failing test cases
#    - Are they near ghosts? What does dispatcher see?
# =========================================================

echo
echo "=== Probe: MMS center-mutation block (mask <= 0) ==="
python3 - << 'EOF'
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy
from tests.helpers.solver_input_schema_dummy import create_validated_input
from src.common.simulation_context import SimulationContext
from src.step3.orchestrate_step3 import orchestrate_step3
from src.step3.boundaries.dispatcher import get_applicable_boundary_configs
from src.common.field_schema import FI

nx = ny = nz = 4
state = make_step2_output_dummy(nx=nx, ny=ny, nz=nz)
context = SimulationContext(input_data=create_validated_input(nx=nx, ny=ny, nz=nz), config=None)

try:
    target_block = next(b for b in state.stencil_matrix if b.center.mask <= 0)
except StopIteration:
    target_block = state.stencil_matrix[0]
    target_block.center.mask = -1

print("Block id:", target_block.id, "mask:", target_block.center.mask)
print("Ghost neighbors (i-/i+/j-/j+/k-/k+):",
      target_block.i_minus.is_ghost, target_block.i_plus.is_ghost,
      target_block.j_minus.is_ghost, target_block.j_plus.is_ghost,
      target_block.k_minus.is_ghost, target_block.k_plus.is_ghost)

rules = get_applicable_boundary_configs(
    target_block,
    state.boundary_conditions.to_dict(),
    state.grid,
    context.input_data.domain_configuration.to_dict()
)
print("DISPATCH RULES:", rules)

target_block.center.set_field(FI.VX_STAR, 1.0)
print("VX_STAR before:", target_block.center.get_field(FI.VX_STAR))
orchestrate_step3(
    block=target_block,
    context=context,
    state_grid=state.grid,
    state_bc_manager=state.boundary_conditions,
    is_first_pass=True,
)
print("VX_STAR after:", target_block.center.get_field(FI.VX_STAR))
EOF

echo
echo "=== Probe: test_gate_3a_3b block (index 500) ==="
python3 - << 'EOF'
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy
from src.step3.boundaries.dispatcher import get_applicable_boundary_configs

state = make_step2_output_dummy(nx=10, ny=10, nz=10)
block = state.stencil_matrix[500]
block.center.mask = -1

boundary_cfg = [{"location": "wall", "type": "no-slip", "values": {"u": 0.1}}]
domain_cfg = {"type": "INTERNAL"}

print("Block id:", block.id, "mask:", block.center.mask)
print("Ghost neighbors (i-/i+/j-/j+/k-/k+):",
      block.i_minus.is_ghost, block.i_plus.is_ghost,
      block.j_minus.is_ghost, block.j_plus.is_ghost,
      block.k_minus.is_ghost, block.k_plus.is_ghost)

try:
    rules = get_applicable_boundary_configs(block, boundary_cfg, state.grid, domain_cfg)
    print("DISPATCH RULES:", rules)
except Exception as e:
    print("RAISED:", repr(e))
EOF

echo
echo "=== Probe: test_gate_3c block (index 10, mask=1, INTERNAL, no cfg) ==="
python3 - << 'EOF'
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy
from src.step3.boundaries.dispatcher import get_applicable_boundary_configs

state = make_step2_output_dummy(nx=4, ny=4, nz=4)
block = state.stencil_matrix[10]
block.center.mask = 1

print("Block id:", block.id, "mask:", block.center.mask)
print("Ghost neighbors (i-/i+/j-/j+/k-/k+):",
      block.i_minus.is_ghost, block.i_plus.is_ghost,
      block.j_minus.is_ghost, block.j_plus.is_ghost,
      block.k_minus.is_ghost, block.k_plus.is_ghost)

try:
    rules = get_applicable_boundary_configs(block, [], state.grid, {"type": "INTERNAL"})
    print("DISPATCH RULES:", rules)
except Exception as e:
    print("RAISED:", repr(e))
EOF

echo
echo "=== Probe: test_gate_3a_missing_wall_config_collision block (index 500) ==="
python3 - << 'EOF'
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy
from src.step3.boundaries.dispatcher import get_applicable_boundary_configs

state = make_step2_output_dummy(nx=10, ny=10, nz=10)
block = state.stencil_matrix[500]
block.center.mask = -1

incomplete_cfg = [
    {"location": "x_min", "type": "no-slip", "values": {"u": 0}},
    {"location": "x_max", "type": "no-slip", "values": {"u": 0}},
    {"location": "y_min", "type": "no-slip", "values": {"v": 0}},
    {"location": "y_max", "type": "no-slip", "values": {"v": 0}},
    {"location": "z_min", "type": "no-slip", "values": {"w": 0}},
    {"location": "z_max", "type": "no-slip", "values": {"w": 0}},
]

print("Block id:", block.id, "mask:", block.center.mask)
print("Ghost neighbors (i-/i+/j-/j+/k-/k+):",
      block.i_minus.is_ghost, block.i_plus.is_ghost,
      block.j_minus.is_ghost, block.j_plus.is_ghost,
      block.k_minus.is_ghost, block.k_plus.is_ghost)

try:
    rules = get_applicable_boundary_configs(block, incomplete_cfg, state.grid, {"type": "INTERNAL"})
    print("DISPATCH RULES:", rules)
except Exception as e:
    print("RAISED:", repr(e))
EOF

# ==========================================
# 3. Sanity: show dispatcher + applier again
# ==========================================
echo
echo "=== dispatcher.py (for reference) ==="
cat -n src/step3/boundaries/dispatcher.py

echo
echo "=== applier.py (for reference) ==="
cat -n src/step3/boundaries/applier.py
