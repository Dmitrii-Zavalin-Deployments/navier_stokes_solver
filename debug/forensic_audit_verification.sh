#!/usr/bin/env bash
# ==============================================================================
# Forensic Audit Script: CMake Missing Source File Investigation & Repair
# ==============================================================================

set -euo pipefail

echo "================================================================================"
echo "[1/3] Diagnostic: Locating CMakeLists.txt and inspecting target definition around line 90"
echo "================================================================================"

if [ -f "CMakeLists.txt" ]; then
    echo "Found CMakeLists.txt. Displaying context lines 75 to 105:"
    cat -n CMakeLists.txt | sed -n '75,105p'
else
    echo "[ERROR] CMakeLists.txt not found in the current working directory!"
    exit 1
fi

echo ""
echo "================================================================================"
echo "[2/3] Smoking-Gun Source Audit: Checking file system presence of target paths"
echo "================================================================================"

TARGET_FILE="cpp/cpp_integration_tests/test_full_pipeline_accelerated_flow.cpp"
if [ -f "$TARGET_FILE" ]; then
    echo "[INFO] Target source file exists: $TARGET_FILE"
    echo "Displaying head of the file:"
    cat -n "$TARGET_FILE" | head -n 20
else
    echo "[SMOKING GUN] Target source file is MISSING: $TARGET_FILE"
    echo "Listing contents of cpp/cpp_integration_tests/ if it exists:"
    ls -la cpp/cpp_integration_tests/ || echo "Directory cpp/cpp_integration_tests/ does not exist."
fi

echo ""
echo "================================================================================"
echo "[3/3] Automated Repair Options (Sed Injections)"
echo "================================================================================"
echo "The following lines show how to remove or comment out the missing target reference."
echo "Uncomment the desired sed command to apply the repair automatically during CI runs."
echo ""

# sed -i '/test_full_pipeline_accelerated_flow\.cpp/d' CMakeLists.txt
# sed -i 's/add_executable(test_full_pipeline_accelerated_flow/# add_executable(test_full_pipeline_accelerated_flow/g' CMakeLists.txt

echo "================================================================================"
echo "Forensic audit complete."
echo "================================================================================"