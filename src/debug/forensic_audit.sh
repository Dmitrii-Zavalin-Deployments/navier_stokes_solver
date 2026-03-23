#!/bin/bash
echo "============================================================"
echo "🔍 SEARCHING FOR THE EXPLOSION (1e10)"
echo "============================================================"

# Search the full log for the 1e10 injection
# This confirms if the BC Applier ever actually SAW the test value
grep "1.0000e+10" simulation.log || echo "❌ CRITICAL: The BC Applier NEVER received the 1e10 value."

# Search for the 'inflow' boundary type in the applier logs
grep "Boundary: inflow" simulation.log | head -n 5

# Check if any BC failed the mapping
grep "CONTRACT VIOLATION" simulation.log