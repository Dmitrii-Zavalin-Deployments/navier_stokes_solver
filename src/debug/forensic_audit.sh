# Define a quick checker function
check_config() {
  local stage=$1
  echo "--- [TRACE: $stage] ---"
  if grep -q "ppe_max_retries" config.json; then
    echo "✅ Key exists."
  else
    echo "❌ KEY LOST! Current content:"
    cat config.json
    exit 1 # Kill the job the moment it disappears
  fi
}