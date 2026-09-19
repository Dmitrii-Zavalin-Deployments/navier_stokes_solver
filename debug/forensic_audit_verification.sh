#!/bin/bash
set -eo pipefail

echo "=== 1. Applying Patch to src/state.py for Initial Conditions Parsing ==="
# Python snippet to insert initial condition population into SolverState.__init__
python3 -c '
path = "src/state.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

target = """        # 4D fields buffer: shape (4, nx, ny, nz) -> [0]: u, [1]: v, [2]: w, [3]: p
        self.fields: np.ndarray = np.zeros((4, self.nx, self.ny, self.nz), dtype=np.float64, order=\"C\")

        # Convenience slices sharing memory views with self.fields
        self.u = self.fields[0]
        self.v = self.fields[1]
        self.w = self.fields[2]
        self.p = self.fields[3]"""

replacement = target + """

        # Populate initial conditions from input_data if present
        init_conds = input_data.get("initial_conditions", {})
        init_vel = init_conds.get("velocity", [0.0, 0.0, 0.0])
        if len(init_vel) >= 3:
            self.u[...] = init_vel[0]
            self.v[...] = init_vel[1]
            self.w[...] = init_vel[2]
        
        init_p = init_conds.get("pressure", 0.0)
        self.p[...] = init_p"""

if target in content and "Populate initial conditions" not in content:
    content = content.replace(target, replacement, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("SUCCESS: Patched src/state.py successfully.")
else:
    print("NOTE: Target already patched or not found.")
'