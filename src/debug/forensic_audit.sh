# Check the recovery logic gate
grep -n "if self.is_in_panic and self._iteration >= 10:" src/common/elasticity.py
# Look for the max_iter reset block
grep -n "self._max_iter = self.config.ppe_max_iter" src/common/elasticity.py

cat -n src/common/elasticity.py