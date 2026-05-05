"""
Example 05 — Suggest experiments: (trans)-β-ionone + solvent + water
=====================================================================
True ternary LLE for β-ionone against each alcohol/ketone solvent in
vanesa.json, with water (4C Aparicio 2007) as diluent.

Solvents
--------
  1-hexanol, 1-octanol, 2-methyl-2-butanol, 2-methyl-3-buten-2-ol,
  2-octanone, mibk
"""

from pathlib import Path

import si_units as si

from pcsaft_quaternary import ternary_lle

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

pure_jsons = [
    str(DATA / "vanesa.json"),
    str(DATA / "water_models.json"),
]

T = 303.15 * si.KELVIN  # 30 °C
P = 1.0 * si.BAR
SOLUTE = "beta-ionone"
DILUENT = "water4C_aparicio2007"

SOLVENTS = [
    "1-hexanol",
    "1-octanol",
    "2-methyl-2-butanol",
    "2-methyl-3-buten-2-ol",
    "2-octanone",
    "mibk",
]

for solvent in SOLVENTS:
    print(f"\n{'=' * 60}")
    print(f"β-ionone + {solvent} + water")
    print("=" * 60)
    try:
        tie_lines, suggestions = ternary_lle(
            pure_json=pure_jsons,
            T=T,
            P=P,
            solute=SOLUTE,
            solvent=solvent,
            diluent=DILUENT,
            induced_association=True,
            output=str(OUT / f"LLE_betaionone+{solvent}+water_T30C"),
            suggest_n=5,
        )
        print(f"  Tie-lines found: {len(tie_lines)}")
        print(f"  Suggested feed compositions ({len(suggestions)} points):")
        print(suggestions)
    except Exception as e:
        print(f"  ERROR: {e}")
