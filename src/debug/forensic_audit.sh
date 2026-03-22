#!/bin/bash

echo "===================================================="
echo "🔍 SIMULATION AUTO-PSY: ROOT CAUSE ANALYSIS"
echo "===================================================="

LOG_FILE="simulation.log"
CONFIG_FILE="config.json"
REPORT="root_cause_report.md"

# 1. TRACE THE ELASTICITY DECAY
echo "Step 1: Analyzing ElasticManager Retries..."
RETRIES=$(grep -c "Instability" $LOG_FILE)
DT_MIN=$(grep "Reducing dt" $LOG_FILE | tail -n 1 | awk '{print $NF}')

# 2. CHECK CONVERGENCE AT TERMINATION
echo "Step 2: Checking Final PPE State..."
LAST_DELTA=$(grep "Non-finite delta" $LOG_FILE | tail -n 1)

# 3. VERIFY CONFIGURATION SSoT
echo "Step 3: Auditing Input Pedigree..."
MAX_ITER=$(grep "ppe_max_iter" $CONFIG_FILE | tr -dc '0-9')
OMEGA=$(grep "ppe_omega" $CONFIG_FILE | awk -F: '{print $2}' | tr -d ' ,')

# 4. GENERATE THE DEFINITIVE REPORT
{
    echo "## 🛑 Root Cause Analysis Report"
    echo "### 1. Numerical Evidence"
    echo "- **Total Stabilization Attempts:** $RETRIES / 10"
    echo "- **Final dt reached:** $DT_MIN"
    
    if [ "$RETRIES" -ge 10 ]; then
        echo "- **Conclusion:** The ElasticManager reached the \`dt_floor\`. The physics is too fast for the current grid spacing."
    fi

    echo "### 2. Convergence Audit"
    if [[ -n "$LAST_DELTA" ]]; then
        echo "- **Failure Mode:** Numerical Explosion (NaN/Inf detected)."
        echo "- **Last Logged Event:** \`$LAST_DELTA\`"
    fi

    echo "### 3. Recommended Fix (Pedigree Update)"
    if [ "$MAX_ITER" -lt 100 ]; then
        echo "- 🟢 **Fix A:** Increase \`ppe_max_iter\` from $MAX_ITER to 200. (Iterations are too low for convergence)."
    else
        echo "- 🟢 **Fix B:** Your iterations ($MAX_ITER) are sufficient. Decrease initial \`time_step\` in input JSON; the simulation is physically jumping too far."
    fi
} > $REPORT

cat $REPORT