"""
Example 04 — Suggest experiments from a pseudoternary LLE diagram
=================================================================
``pseudoternary_lle`` with ``suggest_n > 0`` returns
``(tie_line_data, suggestions)`` in one call.
"""

from pathlib import Path

import si_units as si

from pcsaft_quaternary import pseudoternary_lle

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

pure_jsons = [
    str(DATA / "thiswork2026_pure.json"),
    str(DATA / "water_models.json"),
]

tie_lines, suggestions = pseudoternary_lle(
    pure_json=pure_jsons,
    T=303.15 * si.KELVIN,
    P=1.0 * si.BAR,
    solute="2-phenylethanol",
    solvent1="thymol",
    solvent2="carvacrol",
    diluent="water4C_aparicio2007",
    solvent_ratio=1.0,
    output=str(OUT / "LLE_2PE+thymol+carvacrol+water4C_aparicio_T30C"),
    suggest_n=5,
)

print(f"Tie-lines found: {len(tie_lines)}")
print(suggestions)
