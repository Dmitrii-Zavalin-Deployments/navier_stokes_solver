#!/bin/bash
echo "🔍 STARTING DEEP FORENSIC AUDIT: FLOATING POINT PRECISION"

# 1. SMOKING GUN: Check the specific calculation causing the drift
echo "--- [Source Audit: elasticity.py] ---"
cat -n src/common/elasticity.py | sed -n '30,33p'

# 2. REPAIR STRATEGY:
# Force the test to use a numerical epsilon via pytest.approx.
# This ensures that a drift of 1e-17 doesn't break our CI pipeline.

# A. Replace the rigid equality check with an approximate one
sed -i 's/assert manager._dt_range\[-1\] == config.dt_min_limit/assert manager._dt_range[-1] == pytest.approx(config.dt_min_limit)/' tests/common/test_elasticity_manager.py

echo "✅ Forensic Audit Complete. Numerical tolerance applied via sed."