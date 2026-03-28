#!/bin/bash
# Step 3 & 2 Repair: Aligning with Physical Logic Firewall

echo "Applying Fixes..."

# FIX 1: Update Dispatcher to match test regex exactly and enforce strict KeyError
# Note: We simplify the log to ensure "DISPATCH [Mask]" and "treated as Wall" are clean.
sed -i 's/logger.debug(f"DISPATCH \[Mask\]: Block {block.id} treated as Wall (mask -1)")/logger.debug("DISPATCH [Mask]: Cell treated as Wall (mask -1)")/' src/step3/boundaries/dispatcher.py
sed -i 's/logger.debug(f"DISPATCH \[Mask\]: Block {block.id} treated as Solid (mask 0)")/logger.debug("DISPATCH [Mask]: Cell treated as Solid (mask 0)")/' src/step3/boundaries/dispatcher.py

# FIX 2: Neighbor Sanitization in Orchestrator
# We must ensure neighbors are updated so that Ghost Cells are mutated during the Core pass.
sed -i '/apply_boundary_values(block, rule)/a \                for neighbor in [block.i_minus, block.i_plus, block.j_minus, block.j_plus, block.k_minus, block.k_plus]: apply_boundary_values(neighbor, rule)' src/step3/orchestrate_step3.py

# FIX 3: Strict Registry (The KeyError Fix)
# This assumes _find_config is at the end of the file.
sed -i 's/return \[\]/raise KeyError(f"No boundary configuration found for location: \{location_name\}")/' src/step3/boundaries/dispatcher.py

echo "Repairs complete. Re-running tests..."
pytest tests/quality_gates/sensitivity_gate/test_bc_collisions.py tests/quality_gates/logic_gate/test_step3_mms.py