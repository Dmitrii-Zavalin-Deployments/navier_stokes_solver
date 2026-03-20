# 1. Update main_solver.py to use Config-driven thresholds (Rule 5 Alignment)
# Replace the hardcoded 0.1 with context.input_data.simulation_parameters.min_dt
sed -i '122s/if elasticity.dt < 0.1:/if elasticity.dt < context.input_data.simulation_parameters.min_dt:/' src/main_solver.py

# 2. Update the Test File to include the explicit thresholds
# We will insert min_dt and max_panic_count into the simulation_parameters in the test
sed -i '/"output_interval": 1/s/$/,\n                "min_dt": 0.4/' tests/property_integrity/test_heavy_elasticity_lifecycle.py

# 3. Audit src/common/elasticity.py to ensure it tracks panic events (cat for verification)
cat -n src/common/elasticity.py | head -n 20