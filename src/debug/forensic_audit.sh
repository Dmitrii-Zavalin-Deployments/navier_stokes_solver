#!/bin/bash
# Forensic Audit & Repair (v2 - Indentation Safe)

echo "Applying Indentation-Safe Fixes..."

# Fix the Dispatcher logs (No indentation change needed here)
sed -i 's/logger.debug(f"DISPATCH \[Mask\]: Block {block.id} treated as Wall (mask -1)")/logger.debug("DISPATCH [Mask]: Cell treated as Wall (mask -1)")/' src/step3/boundaries/dispatcher.py

# Fix the Orchestrator with explicit 12-space indentation
# This targets line 60 and adds the neighbor loop below it with correct padding.
sed -i '/apply_boundary_values(block, rule)/a \            for neighbor in [block.i_minus, block.i_plus, block.j_minus, block.j_plus, block.k_minus, block.k_plus]: apply_boundary_values(neighbor, rule)' src/step3/orchestrate_step3.py

echo "Repairs complete. Running tests..."
pytest tests/quality_gates/sensitivity_gate/test_bc_collisions.py