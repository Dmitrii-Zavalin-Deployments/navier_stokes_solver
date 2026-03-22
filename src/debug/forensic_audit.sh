#!/bin/bash

# Configuration
TARGET="config.json"
LOG_FILE="forensic_trace.log"
TIMESTAMP=$(date +"%H:%M:%S")

echo "--- [FORENSIC CHECK @ $TIMESTAMP] ---" | tee -a $LOG_FILE

if [ ! -f "$TARGET" ]; then
    echo "❌ FATAL: $TARGET has been physically DELETED from the disk!" | tee -a $LOG_FILE
    ls -la | tee -a $LOG_FILE
    exit 1
fi

# Check for the key ppe_max_retries
if grep -q "ppe_max_retries" "$TARGET"; then
    echo "✅ Key 'ppe_max_retries' is present." | tee -a $LOG_FILE
    # Show the line for absolute certainty
    grep "ppe_max_retries" "$TARGET" | tee -a $LOG_FILE
else
    echo "❌ CRITICAL: Key 'ppe_max_retries' is MISSING!" | tee -a $LOG_FILE
    echo "Current File Content:" | tee -a $LOG_FILE
    cat "$TARGET" | tee -a $LOG_FILE
    
    # Traceability: Who was the last user/process to touch it?
    echo "File Metadata:" | tee -a $LOG_FILE
    stat "$TARGET" | tee -a $LOG_FILE
fi

echo "--------------------------------------" | tee -a $LOG_FILE