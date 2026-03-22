#!/bin/bash

LABEL=${1:-"Unknown Step"}
FILE="config.json"

echo -e "\n--- [AUDIT: $LABEL] ---"

# 1. Check if file exists
if [ ! -f "$FILE" ]; then
    echo "❌ FATAL: $FILE NOT FOUND."
    echo "Current Dir: $(pwd)"
    ls -la
    exit 0 # We don't exit 1 yet so we can see other steps
fi

# 2. Read and Display Raw Content
RAW_CONTENT=$(cat "$FILE")
echo "📄 Raw Content: $RAW_CONTENT"

# 3. Check for the specific key
# We use -q (quiet) because we just want the exit status
if grep -q "\"ppe_max_retries\"" "$FILE"; then
    # Extract value for confirmation
    VALUE=$(grep -oP '"ppe_max_retries":\s*\K[0-9]+' "$FILE")
    echo "✅ PASS: 'ppe_max_retries' is present (Value: $VALUE)"
else
    echo "❌ FAIL: 'ppe_max_retries' KEY IS MISSING!"
    
    # Debug: Show what keys ARE there by stripping values and braces
    KEYS=$(sed 's/[:].*//g; s/[{}"]//g; s/^[[:space:]]*//' "$FILE" | tr '\n' ',' | sed 's/,$//')
    echo "Available Keys: [$KEYS]"
fi

echo "--------------------------------"