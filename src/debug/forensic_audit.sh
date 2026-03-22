#!/bin/bash
STEP_NAME=$1
FILE="config.json"

echo "--- [CHECKPOINT: $STEP_NAME] ---"
if [ -f "$FILE" ]; then
    echo "✅ $FILE exists."
    echo "   Permissions: $(stat -c '%a' $FILE)"
    echo "   Size: $(stat -c '%s' bytes $FILE)"
    echo "   First 2 lines: $(head -n 2 $FILE | tr -d '\n')"
else
    echo "❌ ERROR: $FILE is MISSING at this stage!"
    echo "   Current Directory: $(pwd)"
    echo "   Directory Content: $(ls -m)"
    
    # Check if it was accidentally moved to a subfolder
    FOUND_PATH=$(find . -name "$FILE")
    if [ -n "$FOUND_PATH" ]; then
        echo "   🔍 FOUND AT WRONG PATH: $FOUND_PATH"
    fi
    exit 1
fi
echo "--------------------------------"