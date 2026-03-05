"""
Example 01 — Pseudoternary LLE: 2-phenylethanol + (thymol:carvone 1:1) + water
================================================================================
System
------
  Solute   : 2-phenylethanol
  Solvents : thymol + carvone  (equimolar pseudo-solvent, solvent_ratio = 1.0)
  Diluent  : water  (Cameretti 2008 PC-SAFT)

Parameters spread across two JSON files:
  - thiswork2026_pure.json  → 2-phenylethanol, thymol, carvone
  - water_models.json       → water_cameretti2008

No binary interaction parameters (k_ij = 0).
"""

from pathlib import Path

import si_units as si

from pcsaft_quaternary import pseudoternary_lle

# --- paths (relative to this script) ---
DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

pure_jsons = [
    str(DATA / "thiswork2026_pure.json"),
    str(DATA / "water_models.json"),
]

# --- run ---
tie_lines = pseudoternary_lle(
    pure_json=pure_jsons,
    T=298.15 * si.KELVIN,
    P=1.0 * si.BAR,
    solute="2-phenylethanol",
    solvent1="camphor",
    solvent2="carvacrol",
    diluent="water_cameretti2008",
    solvent_ratio=2.0,
    output=str(OUT / "LLE_2PE+thymol+carvone+water_T25C_P1bar.png"),
)

print(f"\nTotal LLE tie-lines found: {len(tie_lines)}")
