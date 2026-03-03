"""
Example 02 — Pseudoternary LLE: furfuryl alcohol + (thymol:octanoic acid 1:1) + water
========================================================================================
System
------
  Solute   : furfuryl alcohol
  Solvents : thymol + octanoic acid  (equimolar pseudo-solvent, solvent_ratio = 1.0)
  Diluent  : water  (Cameretti 2008 PC-SAFT)
  T        : 40 °C

Parameters spread across two JSON files:
  - ceaklapp_pure.json  → furfuryl_alcohol_ceaklapp, thymol_ceaklapp, octanoic_acid_ceaklapp
  - water_models.json   → water_cameretti2008

No binary interaction parameters (k_ij = 0).
"""

from pathlib import Path

import si_units as si

from pcsaft_quaternary import pseudoternary_lle

# --- paths (relative to this script) ---
DATA = Path(__file__).parent / "data"
OUT  = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

pure_jsons = [
    str(DATA / "ceaklapp_pure.json"),
    str(DATA / "water_models.json"),
]

# --- run ---
tie_lines = pseudoternary_lle(
    pure_json=pure_jsons,
    T=313.15 * si.KELVIN,
    P=1.0 * si.BAR,
    solute="furfuryl_alcohol_ceaklapp",
    solvent1="thymol_ceaklapp",
    solvent2="octanoic_acid_ceaklapp",
    diluent="water_cameretti2008",
    solvent_ratio=1.0,
    output=str(OUT / "LLE_furfuryl+thymol+octanoic+water_T40C_P1bar.png"),
)

print(f"\nTotal LLE tie-lines found: {len(tie_lines)}")
