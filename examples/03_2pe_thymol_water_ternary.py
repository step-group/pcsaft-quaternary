"""
Example 03 — True ternary LLE: 2-phenylethanol + carvone + water
=================================================================
System
------
  Solute  : 2-phenylethanol
  Solvent : carvone
  Diluent : water  (Rehner 2020 4C PC-PSAFT)

Parameters spread across two JSON files:
  - thiswork2026_pure.json  → 2-phenylethanol, carvone
  - water_models.json       → water_4C_pcpsaft_rehner2020

No binary interaction parameters (k_ij = 0).
"""

from pathlib import Path

import si_units as si

from pcsaft_quaternary import ternary_lle

# --- paths (relative to this script) ---
DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

pure_jsons = [
    str(DATA / "thiswork2026_pure.json"),
    str(DATA / "water_models.json"),
]

# --- run ---
tie_lines = ternary_lle(
    pure_json=pure_jsons,
    T=298.15 * si.KELVIN,
    P=1.0 * si.BAR,
    solute="2-phenylethanol",
    solvent="camphor",
    diluent="water_4C_pcpsaft_rehner2020",
    output=str(OUT / "LLE_2PE+camphor+water_T25C_P1bar.png"),
    induced_association=True,
)

print(f"\nTotal LLE tie-lines found: {len(tie_lines)}")
