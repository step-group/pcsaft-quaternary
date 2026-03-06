"""pcsaft-quaternary: pseudoternary LLE phase diagrams via PC-SAFT (feos)."""

import warnings
from pathlib import Path

import si_units as si

from .lle import (
    _to_pseudo_ternary_mass,
    _to_ternary_mass,
    build_eos,
    scan_pseudoternary,
    scan_ternary,
    suggest_experiments,
)
from .plot import plot_pseudoternary_lle


def pseudoternary_lle(
    pure_json,
    T,
    P,
    solute,
    solvent1,
    solvent2,
    diluent,
    solvent_ratio,
    binary_json=None,
    output=None,
    n_points=51,
    solvent_label=None,
    mass_basis=True,
    induced_association=True,
    exp_tie_lines=None,
    suggest_n=0,
    suggest_phi=0.5,
):
    """Compute and plot a pseudoternary LLE phase diagram using PC-SAFT.

    Two of the four components (``solvent1`` and ``solvent2``) are combined into a
    single "pseudo-solvent" axis so the four-component system can be displayed on a
    standard ternary diagram.

    Parameters
    ----------
    pure_json : str or list[str]
        Path(s) to JSON file(s) containing PC-SAFT pure-component parameters.
        Use a list when parameters are spread across multiple files.
    T : si_units.SIObject
        Temperature, e.g. ``298.15 * si.KELVIN``.
    P : si_units.SIObject
        Pressure, e.g. ``101325.0 * si.PASCAL``.
    solute : str
        Name of the solute component (must match an entry in ``pure_json``).
    solvent1 : str
        Name of the first solvent component.
    solvent2 : str
        Name of the second solvent component.
    diluent : str
        Name of the diluent component.
    solvent_ratio : float
        Molar ratio of ``solvent1`` to ``solvent2`` (n1/n2).
        Use ``1.0`` for equimolar, ``2.0`` for 2:1, ``0.5`` for 1:2, etc.
    binary_json : str or None
        Optional path to a JSON file with binary interaction parameters (k_ij).
    output : str or None
        Output file base name (without extension).  If ``None`` an auto-generated
        name is used.  The ``.pdf`` extension is always appended automatically.
    n_points : int
        Number of grid divisions along each pseudo-ternary axis (default 51).
        Higher values give a denser scan but take longer.
    solvent_label : str or None
        Custom axis label for the pseudo-solvent axis.  Defaults to
        ``"<solvent1>+<solvent2> (<ratio>:1)"``, e.g. ``"thymol+geraniol (1:1)"``.
    mass_basis : bool
        If ``True`` (default), plot and return pseudo-ternary coordinates in
        mass fractions.  If ``False``, use mole fractions.
    induced_association : bool or list[str]
        If ``True``, apply induced association to every non-associating
        component except the diluent.  If a list of component names, apply
        only to those specific components.  The diluent's kappa_ab is used
        as the reference; epsilon_k_ab is forced to 0.0.
    exp_tie_lines : list[dict] or None
        Optional experimental tie-lines to overlay on the diagram. Each dict
        must have keys ``phase1_pseudo`` and ``phase2_pseudo`` (3-tuples:
        solute, pseudo-solvent, diluent) in mole fractions.
    suggest_n : int
        If > 0, run ``suggest_experiments`` and return
        ``(tie_line_data, suggestions)`` instead of just ``tie_line_data``.
    suggest_phi : float
        Target volumetric fraction of phase 1 for the suggested experiments
        (default 0.5).

    Returns
    -------
    list[dict] or tuple[list[dict], list[dict]]
        When ``suggest_n == 0``: one dict per detected LLE tie-line.
        When ``suggest_n > 0``: ``(tie_line_data, suggestions)`` where
        ``suggestions`` is the output of ``suggest_experiments``.

    Side effects
    ------------
    Saves a PDF to ``output`` (or the auto-generated path).
    """
    # Convert molar ratio r = n1/n2 → mole fraction frac1 = r / (1 + r)
    frac1 = solvent_ratio / (1.0 + solvent_ratio)

    # Extract plain floats for labels / filename
    T_K = float(T / si.KELVIN)
    P_Pa = float(P / si.PASCAL)

    component_names = [solute, solvent1, solvent2, diluent]

    # Build EOS (also returns molar masses for optional mass-fraction conversion)
    eos, molar_masses = build_eos(
        pure_json, component_names, binary_json=binary_json,
        induced_association=induced_association,
    )

    # Scan ternary grid (always returns mole-fraction data in *_4comp fields)
    tie_line_data = scan_pseudoternary(
        eos, T, P, solvent_ratio=frac1, n_points=n_points
    )

    # Reproject pseudo-ternary coordinates to mass fractions if requested
    if mass_basis:
        for d in tie_line_data:
            d["feed_pseudo"] = _to_pseudo_ternary_mass(d["feed_4comp"], molar_masses)
            d["phase1_pseudo"] = _to_pseudo_ternary_mass(
                d["phase1_4comp"], molar_masses
            )
            d["phase2_pseudo"] = _to_pseudo_ternary_mass(
                d["phase2_4comp"], molar_masses
            )

    # Axis labels
    if solvent_label is not None:
        ps_label = solvent_label
    else:
        r_str = f"{solvent_ratio:.4g}"
        ps_label = f"{solvent1}+{solvent2} ({r_str}:1)"
    names_pseudo = [solute, ps_label, diluent]

    # Auto-generate output path
    if output is None:
        T_C = T_K - 273.15
        P_bar = P_Pa / 1e5
        output = (
            f"LLE_{solute}+{solvent1}+{solvent2}+{diluent}"
            f"_T{T_C:.0f}C_P{P_bar:.3f}bar"
        )
    elif Path(output).suffix:
        warnings.warn(
            f"output extension '{Path(output).suffix}' ignored; saving as .pdf",
            UserWarning,
            stacklevel=2,
        )
    output = str(Path(output).with_suffix(".pdf"))

    # Compute suggestions before plotting so the points appear on the main diagram
    suggestions = None
    if suggest_n > 0:
        suggestions = suggest_experiments(
            tie_line_data, eos, T, P,
            n=suggest_n,
            target_phi=suggest_phi,
            mass_basis=mass_basis,
            molar_masses=molar_masses,
        )

    # Plot (suggested points overlaid when present)
    plot_pseudoternary_lle(
        tie_line_data, names_pseudo, T_K, P_Pa, output,
        exp_tie_lines=exp_tie_lines,
        suggested_points=suggestions,
    )

    if suggest_n > 0:
        return tie_line_data, suggestions

    return tie_line_data


def ternary_lle(
    pure_json,
    T,
    P,
    solute,
    solvent,
    diluent,
    binary_json=None,
    output=None,
    n_points=51,
    mass_basis=True,
    induced_association=True,
    exp_tie_lines=None,
    suggest_n=0,
    suggest_phi=0.5,
):
    """Compute and plot a true ternary LLE phase diagram using PC-SAFT.

    Parameters
    ----------
    pure_json : str or list[str]
        Path(s) to JSON file(s) containing PC-SAFT pure-component parameters.
    T : si_units.SIObject
        Temperature, e.g. ``298.15 * si.KELVIN``.
    P : si_units.SIObject
        Pressure, e.g. ``101325.0 * si.PASCAL``.
    solute : str
        Name of the solute component (plotted on the right axis).
    solvent : str
        Name of the solvent component (plotted on the left axis).
    diluent : str
        Name of the diluent component (plotted on the bottom axis).
    binary_json : str or None
        Optional path to a JSON file with binary interaction parameters (k_ij).
    output : str or None
        Output file base name (without extension).  If ``None`` an auto-generated
        name is used.  The ``.pdf`` extension is always appended automatically.
    n_points : int
        Number of grid divisions along each ternary axis (default 51).
    mass_basis : bool
        If ``True`` (default), plot and return coordinates in mass fractions.
        If ``False``, use mole fractions.
    induced_association : bool or list[str]
        Same semantics as in ``pseudoternary_lle``.
    exp_tie_lines : list[dict] or None
        Optional experimental tie-lines to overlay. Same semantics as in
        ``pseudoternary_lle``.
    suggest_n : int
        If > 0, run ``suggest_experiments`` and return
        ``(tie_line_data, suggestions)`` instead of just ``tie_line_data``.
    suggest_phi : float
        Target volumetric fraction of phase 1 for the suggested experiments
        (default 0.5).

    Returns
    -------
    list[dict] or tuple[list[dict], list[dict]]
        When ``suggest_n == 0``: one dict per detected LLE tie-line.
        When ``suggest_n > 0``: ``(tie_line_data, suggestions)``.

    Side effects
    ------------
    Saves a PDF to ``output`` (or the auto-generated path).
    """
    T_K = float(T / si.KELVIN)
    P_Pa = float(P / si.PASCAL)

    component_names = [solute, solvent, diluent]

    eos, molar_masses = build_eos(
        pure_json, component_names, binary_json=binary_json,
        induced_association=induced_association,
    )

    tie_line_data = scan_ternary(eos, T, P, n_points=n_points)

    if mass_basis:
        for d in tie_line_data:
            d["feed_pseudo"]   = _to_ternary_mass(d["feed_3comp"],   molar_masses)
            d["phase1_pseudo"] = _to_ternary_mass(d["phase1_3comp"], molar_masses)
            d["phase2_pseudo"] = _to_ternary_mass(d["phase2_3comp"], molar_masses)

    if output is None:
        T_C = T_K - 273.15
        P_bar = P_Pa / 1e5
        output = f"LLE_{solute}+{solvent}+{diluent}_T{T_C:.0f}C_P{P_bar:.3f}bar"
    elif Path(output).suffix:
        warnings.warn(
            f"output extension '{Path(output).suffix}' ignored; saving as .pdf",
            UserWarning,
            stacklevel=2,
        )
    output = str(Path(output).with_suffix(".pdf"))

    suggestions = None
    if suggest_n > 0:
        suggestions = suggest_experiments(
            tie_line_data, eos, T, P,
            n=suggest_n,
            target_phi=suggest_phi,
            mass_basis=mass_basis,
            molar_masses=molar_masses,
        )

    plot_pseudoternary_lle(
        tie_line_data, component_names, T_K, P_Pa, output,
        exp_tie_lines=exp_tie_lines,
        suggested_points=suggestions,
    )

    if suggest_n > 0:
        return tie_line_data, suggestions

    return tie_line_data
