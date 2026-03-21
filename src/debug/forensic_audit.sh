# 1. Audit the Recovery Logic: Line 85 is the crash site
cat -n src/common/elasticity.py | sed -n '80,90p'

# 2. Verify Config Slots: Confirm 'dt' is indeed gone (as intended)
grep -A 5 "__slots__" src/common/solver_config.py