"""Overlay MARTINI MD tie-lines (Status='OK') on the PC-SAFT pseudoternary diagrams.

The 'Number Density' sheet is in MARTINI beads/nm^3, so bead counts are divided by
BEADS before mass fractions are formed.  Bead counts come from
MT-4/martini_DES/martini_DES/ff/*.itp; one W bead represents 4 water molecules.
"""

from collections import defaultdict
from pathlib import Path

import openpyxl
import si_units as si

from pcsaft_quaternary import pseudoternary_lle

ROOT = Path(__file__).parent.parent
DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "out"
XLSX = ROOT / "tie_lines-MARTINI.xlsx"

# beads per molecule (ff/*.itp); water is the inverse: 1 W bead == 4 molecules
BEADS = {
    "camphor": 3, "carvone": 4, "carvacrol": 4, "eugenol": 5,
    "thymol": 4, "geraniol": 4, "2-phenylethanol": 4, "water": 0.25,
}
MW = {
    "camphor": 152.24, "carvone": 150.22, "carvacrol": 150.22, "eugenol": 164.20,
    "thymol": 150.22, "geraniol": 154.25, "2-phenylethanol": 122.17, "water": 18.02,
}

# short name used in the existing out/ filenames, keyed by (HBA, HBD)
SYSTEM_NAMES = {
    ("thymol", "carvone"): "ThyCar", ("thymol", "geraniol"): "ThyGer",
    ("thymol", "carvacrol"): "ThyCarvac", ("thymol", "eugenol"): "ThyEug",
    ("thymol", "camphor"): "ThyCam", ("camphor", "carvone"): "CamCar",
    ("camphor", "geraniol"): "CamGer", ("camphor", "carvacrol"): "CamCarvac",
    ("camphor", "eugenol"): "CamEug",
}

SOLUTE = "2-phenylethanol"
DILUENT = "water4C_aparicio2007"


def _mass_pseudo(dens, hba, hbd):
    """{component: molecular density} -> (w_solute, w_pseudo_solvent, w_diluent).

    Mass fractions are ratios, so any consistent density unit works (mol/L,
    molecules/nm^3, …) as long as every component uses the same one.
    """
    mass = {k: v * MW[k] for k, v in dens.items()}
    total = sum(mass.values())
    return (
        mass[SOLUTE] / total,
        (mass[hba] + mass[hbd]) / total,
        mass["water"] / total,
    )


def _from_beads(n_hba, n_hbd, n_water, n_phe, hba, hbd):
    """MARTINI beads/nm^3 -> pseudo-ternary mass fractions."""
    return _mass_pseudo({
        hba: n_hba / BEADS[hba],
        hbd: n_hbd / BEADS[hbd],
        "water": n_water / BEADS["water"],
        SOLUTE: n_phe / BEADS[SOLUTE],
    }, hba, hbd)


def _from_mol_per_l(c_hba, c_hbd, c_water, c_phe, hba, hbd):
    """Molecular molar densities (mol/L) -> pseudo-ternary mass fractions."""
    return _mass_pseudo(
        {hba: c_hba, hbd: c_hbd, "water": c_water, SOLUTE: c_phe}, hba, hbd
    )


def read_md_tie_lines(path=XLSX):
    """{(hba, hbd): [{'phase1_pseudo':…, 'phase2_pseudo':…}, …]} for Status == 'OK'.

    Rows present in the 'Mol Density' sheet are recomputed molecular densities and
    take precedence; everything else falls back to the raw MARTINI bead counts.
    """
    wb = openpyxl.load_workbook(path, data_only=True)
    molar = {}
    if "Mol Density" in wb.sheetnames:
        molar = {
            r[0]: r
            for r in wb["Mol Density"].iter_rows(min_row=4, values_only=True)
            if r[0] is not None
        }

    out = defaultdict(list)
    for row in wb["Number Density"].iter_rows(min_row=4, max_row=27, values_only=True):
        if row[6] != "OK":
            continue
        hba, hbd = row[4], row[5]
        src, conv = (molar[row[0]], _from_mol_per_l) if row[0] in molar else (row, _from_beads)
        out[(hba, hbd)].append({
            "phase1_pseudo": conv(*src[7:11], hba, hbd),    # DES-rich
            "phase2_pseudo": conv(*src[12:16], hba, hbd),   # water-rich
        })
    return dict(out)


def main(only=None):
    pure_jsons = [str(DATA / "thiswork2026_pure.json"), str(DATA / "water_models.json")]
    md = read_md_tie_lines()

    for (hba, hbd), tls in md.items():
        name = SYSTEM_NAMES[(hba, hbd)]
        if only and name not in only:
            continue
        print(f"\n{name}: {hba} + {hbd} — {len(tls)} MD tie-line(s)")
        pseudoternary_lle(
            pure_json=pure_jsons,
            T=303.15 * si.KELVIN,
            P=1.0 * si.BAR,
            solute=SOLUTE,
            solvent1=hba,
            solvent2=hbd,
            diluent=DILUENT,
            solvent_ratio=1.0,
            mass_basis=True,
            exp_tie_lines=tls,
            output=str(OUT / f"LLE_2PE+{name}+water_T30C_MD"),
        )


def _self_check():
    md = read_md_tie_lines()
    assert sum(len(v) for v in md.values()) == 13, "expected 13 OK tie-lines"
    assert set(md) == {
        ("camphor", "carvone"), ("camphor", "carvacrol"), ("camphor", "eugenol"),
        ("thymol", "camphor"), ("thymol", "carvone"), ("thymol", "geraniol"),
    }
    for tls in md.values():
        for tl in tls:
            for key in ("phase1_pseudo", "phase2_pseudo"):
                assert abs(sum(tl[key]) - 1.0) < 1e-9, f"{key} does not sum to 1"
            # water-rich phase must actually be water-rich
            assert tl["phase2_pseudo"][2] > 0.9
            assert tl["phase1_pseudo"][1] > 0.9   # DES-rich phase is DES-rich

    # thymol:camphor comes from the recomputed 'Mol Density' rows
    thycam = md[("thymol", "camphor")]
    got = sorted(round(tl["phase1_pseudo"][0], 5) for tl in thycam)
    assert got == [0.01054, 0.04163, 0.08233], f"unexpected ThyCam w(2PE): {got}"


if __name__ == "__main__":
    import sys

    _self_check()
    main(only=sys.argv[1:] or None)
