# src/step4/boundary_applier.py

def apply_boundary_values(block, rule: dict):
    """
    Selective update: only updates attributes present in the rule['values'].
    Maps schema keys (u, v, w) to Cell attributes (vx, vy, vz).
    """
    values = rule.get("values", {})
    
    if "u" in values: 
        block.center.vx = values["u"]
    if "v" in values: 
        block.center.vy = values["v"]
    if "w" in values: 
        block.center.vz = values["w"]
    if "p" in values: 
        block.center.p = values["p"]