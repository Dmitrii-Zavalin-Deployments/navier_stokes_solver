# --- src/debug/forensic_audit.sh ---
#!/bin/bash

echo "🔍 DIAGNOSTIC: Comparing Input Factory vs Step 1 Dummy Structure"
# Check what the input factory actually produces for inflow
python3 -c "from tests.helpers.solver_input_schema_dummy import create_validated_input; print('INPUT BC[0]:', create_validated_input(nx=4,ny=4,nz=4).boundary_conditions.items[0].values)"
# Check what the dummy thinks it should look like
python3 -c "from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy; print('DUMMY BC[0]:', make_step1_output_dummy(nx=4,ny=4,nz=4).boundary_conditions.to_dict()['items'][0]['values'])"

echo -e "\n📂 SMOKING-GUN AUDIT: tests/helpers/solver_step1_output_dummy.py"
cat -n tests/helpers/solver_step1_output_dummy.py | grep -A 5 "x_min"

echo -e "\n🛠️  AUTOMATED REPAIR: Aligning Dummy Factories with Input Schema"
# The fix: Remove the 'p': 0.0 or 'p': 1.0 from Inflow/No-Slip dictionaries in the dummies 
# where they shouldn't exist according to the validated input schema.

# Repair Step 1 Dummy
sed -i "s/'u': 1.0, 'v': 0.0, 'w': 0.0, 'p': [0-9.]*/'u': 1.0, 'v': 0.0, 'w': 0.0/g" tests/helpers/solver_step1_output_dummy.py
sed -i "s/'u': 0.0, 'v': 0.0, 'w': 0.0, 'p': [0-9.]*/'u': 0.0, 'v': 0.0, 'w': 0.0/g" tests/helpers/solver_step1_output_dummy.py

# Repair Step 2 Dummy (Inherits the same structure)
sed -i "s/'u': 1.0, 'v': 0.0, 'w': 0.0, 'p': [0-9.]*/'u': 1.0, 'v': 0.0, 'w': 0.0/g" tests/helpers/solver_step2_output_dummy.py

echo -e "\n✅ Audit Complete. Structural parity should be restored."