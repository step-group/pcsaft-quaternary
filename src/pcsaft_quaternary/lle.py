"""Core PC-SAFT flash calculation logic for pseudoternary LLE scanning."""

import json

import numpy as np
import si_units as si
from feos import (
    BinaryRecord,
    EquationOfState,
    IdentifierOption,
    Parameters,
    PhaseEquilibrium,
    PureRecord,
)


def _components_in_file(path, component_names):
    """Return the subset of component_names whose 'name' identifier appears in a JSON file."""
    with open(path) as f:
        data = json.load(f)
    names_in_file = {entry["identifier"]["name"] for entry in data}
    return [c for c in component_names if c in names_in_file]


def _is_non_associating(record):
    """Return True if a PureRecord has no site with both na >= 1 and nb >= 1."""
    for site in record.association_sites:
        if site.get("na", 0.0) >= 1.0 and site.get("nb", 0.0) >= 1.0:
            return False
    return True


def _load_binary_records(binary_json):
    """Load a binary-interaction JSON file into a list of BinaryRecord objects."""
    with open(binary_json) as f:
        data = json.load(f)
    return [BinaryRecord.from_json_str(json.dumps(entry)) for entry in data]


def _apply_induced_association(pure_records, diluent_idx, target_names):
    """Return a new list of PureRecords with induced association applied.

    For each component in *target_names* that is currently non-associating
    (no site has both na >= 1 and nb >= 1), an association site is added with:

    - na = 1, nb = 1
    - kappa_ab  = kappa_ab of the diluent
    - epsilon_k_ab = 0.0

    Parameters
    ----------
    pure_records : list[PureRecord]
        Records in component order.
    diluent_idx : int
        Index of the diluent component (provides kappa_ab reference).
    target_names : set[str]
        Component names to consider for modification.

    Returns
    -------
    list[PureRecord]
        Modified records (originals are left unchanged; only non-associating
        targets receive the new site).

    Raises
    ------
    ValueError
        If the diluent has no association site with kappa_ab defined.
    """
    # --- get diluent kappa_ab ---
    diluent_sites = pure_records[diluent_idx].association_sites
    kappa_ref = None
    for site in diluent_sites:
        if "kappa_ab" in site:
            kappa_ref = site["kappa_ab"]
            break
    if kappa_ref is None:
        diluent_name = pure_records[diluent_idx].identifier.name
        raise ValueError(
            f"Induced association requires the diluent ({diluent_name!r}) to have "
            "a kappa_ab parameter, but none was found."
        )

    induced_site = {"na": 1.0, "nb": 1.0, "kappa_ab": kappa_ref, "epsilon_k_ab": 0.0}

    modified = []
    for record in pure_records:
        name = record.identifier.name
        if name in target_names and _is_non_associating(record):
            mr = record.model_record
            record = PureRecord(
                record.identifier,
                record.molarweight,
                association_sites=[induced_site],
                **mr,
            )
        modified.append(record)
    return modified


def build_eos(pure_json, component_names, binary_json=None, induced_association=False):
    """Load PC-SAFT parameters from JSON file(s) and return a feos EOS object.

    Parameters
    ----------
    pure_json : str or list[str]
        Path(s) to JSON file(s) containing pure-component PC-SAFT parameters.
    component_names : list[str]
        Names of the four components in order [solute, solvent1, solvent2, diluent].
    binary_json : str or None
        Optional path to JSON file containing binary interaction parameters (kij).
    induced_association : bool or list[str]
        If ``True``, apply induced association to every non-associating component
        except the diluent (last entry in *component_names*).
        If a list of component names, apply only to those specific components.
        In both cases the diluent's kappa_ab is used as the reference; its
        epsilon_k_ab is set to 0.0.  Non-associating is defined as: no
        association site with **both** na >= 1 **and** nb >= 1.

    Returns
    -------
    eos : feos.EquationOfState
    molar_masses : np.ndarray
        Molar masses in g/mol, in the same order as ``component_names``.
    """
    if isinstance(pure_json, (list, tuple)):
        input_pairs = [
            (found, path)
            for path in pure_json
            if (found := _components_in_file(path, component_names))
        ]
        params = Parameters.from_multiple_json(
            input=input_pairs,
            binary_path=binary_json,
            identifier_option=IdentifierOption.Name,
        )
    else:
        params = Parameters.from_json(
            substances=component_names,
            pure_path=pure_json,
            binary_path=binary_json,
            identifier_option=IdentifierOption.Name,
        )

    # Induced association: modify pure records and rebuild params.
    if induced_association is not False and induced_association is not None:
        diluent_idx = len(component_names) - 1
        if induced_association is True:
            # All non-diluent components are candidates.
            target_names = set(component_names[:diluent_idx])
        else:
            target_names = set(induced_association)

        pure_records = list(params.pure_records)
        pure_records = _apply_induced_association(pure_records, diluent_idx, target_names)
        binary_records = _load_binary_records(binary_json) if binary_json else []
        params = Parameters.from_records(
            pure_records,
            binary_records=binary_records,
            identifier_option=IdentifierOption.Name,
        )

    molar_masses = np.array([r.molarweight for r in params.pure_records])
    return EquationOfState.pcsaft(params), molar_masses


def _to_pseudo_ternary(x4):
    """Project 4-component mole fraction vector to pseudo-ternary coordinates.

    Component ordering: [solute, solvent1, solvent2, diluent]
    Pseudo-ternary: (solute, pseudo-solvent = solvent1 + solvent2, diluent)
    """
    return (float(x4[0]), float(x4[1]) + float(x4[2]), float(x4[3]))


def _to_pseudo_ternary_mass(x4, M):
    """Project 4-component mole fractions to pseudo-ternary mass-fraction coordinates.

    Parameters
    ----------
    x4 : array-like, shape (4,)
        Mole fractions [solute, solvent1, solvent2, diluent].
    M : array-like, shape (4,)
        Molar masses in g/mol [M_solute, M_solv1, M_solv2, M_diluent].

    Returns
    -------
    (w_solute, w_pseudo_solvent, w_diluent) : tuple of float
        Normalised mass fractions summing to 1.
    """
    w = np.asarray(x4) * np.asarray(M)
    w = w / w.sum()
    return (float(w[0]), float(w[1]) + float(w[2]), float(w[3]))


def _to_ternary_mass(x3, M):
    """Convert 3-component mole fractions to mass fractions.

    Parameters
    ----------
    x3 : array-like, shape (3,)
        Mole fractions [A, B, C].
    M : array-like, shape (3,)
        Molar masses in g/mol.

    Returns
    -------
    (w_A, w_B, w_C) : tuple of float
        Normalised mass fractions summing to 1.
    """
    w = np.asarray(x3) * np.asarray(M)
    w = w / w.sum()
    return (float(w[0]), float(w[1]), float(w[2]))


def scan_ternary(eos, T, P, n_points=51):
    """Scan a triangular grid of feed compositions and detect LLE tie-lines
    for a true 3-component system.

    The three components are assumed to be in the order used when building the
    EOS: [A (0), B (1), C (2)].

    Parameters
    ----------
    eos : feos.EquationOfState
        PC-SAFT equation of state for the three-component system.
    T : si_units.SIObject
        Temperature, e.g. ``298.15 * si.KELVIN``.
    P : si_units.SIObject
        Pressure, e.g. ``101325.0 * si.PASCAL``.
    n_points : int
        Number of grid divisions along each ternary axis.

    Returns
    -------
    list[dict]
        Each entry contains:
        - ``phase1_pseudo`` : 3-tuple  (x_A, x_B, x_C) of the liquid-rich phase
        - ``phase2_pseudo`` : 3-tuple  (x_A, x_B, x_C) of the second liquid phase
        - ``feed_pseudo``   : 3-tuple  (x_A, x_B, x_C) of the feed
        - ``phase1_3comp``  : np.ndarray  3-component mole fractions of phase 1
        - ``phase2_3comp``  : np.ndarray  3-component mole fractions of phase 2
        - ``feed_3comp``    : np.ndarray  3-component mole fractions of the feed

    Notes
    -----
    The ``phase1_pseudo`` / ``phase2_pseudo`` / ``feed_pseudo`` keys intentionally
    mirror the output of ``scan_pseudoternary`` so that
    ``plot_pseudoternary_lle`` can be reused unchanged.
    """
    results = []
    n = n_points

    for i in range(0, n + 1):        # component A
        for j in range(0, n - i + 1):  # component B
            k = n - i - j              # component C
            # Skip pure-component corners
            if (i == 0) + (j == 0) + (k == 0) >= 2:
                continue

            feed = np.array([i / n, j / n, k / n], dtype=float)
            feed = feed / feed.sum()

            try:
                result = PhaseEquilibrium.tp_flash(
                    eos, T, P, feed * si.MOL, max_iter=10000
                )
            except Exception:
                continue

            x1 = np.array(result.liquid.molefracs)
            x2 = np.array(result.vapor.molefracs)

            _mol_m3 = si.MOL / si.METER**3
            rho1 = result.liquid.density / _mol_m3
            rho2 = result.vapor.density / _mol_m3
            if rho1 <= 600 or rho2 <= 600:
                continue

            if np.allclose(x1, x2, atol=1e-4):
                continue

            results.append(
                {
                    "feed_pseudo":   tuple(float(v) for v in feed),
                    "phase1_pseudo": tuple(float(v) for v in x1),
                    "phase2_pseudo": tuple(float(v) for v in x2),
                    "feed_3comp":    feed,
                    "phase1_3comp":  x1,
                    "phase2_3comp":  x2,
                }
            )

    return results


def scan_pseudoternary(eos, T, P, solvent_ratio, n_points=51):
    """Scan a triangular grid of feed compositions and detect LLE tie-lines.

    The four components are assumed to be in the order used when building the EOS:
    [solute (0), solvent1 (1), solvent2 (2), diluent (3)].

    Parameters
    ----------
    eos : feos.EquationOfState
        PC-SAFT equation of state for the four-component system.
    T : si_units.SIObject
        Temperature, e.g. ``298.15 * si.KELVIN``.
    P : si_units.SIObject
        Pressure, e.g. ``101325.0 * si.PASCAL``.
    solvent_ratio : float
        Mole fraction of solvent1 within the pseudo-solvent (solvent1 + solvent2).
        Must be in (0, 1).
    n_points : int
        Number of grid divisions along each pseudo-ternary axis.

    Returns
    -------
    list[dict]
        Each entry contains:
        - ``feed_pseudo``   : 3-tuple  (solute, pseudo-solvent, diluent) for feed
        - ``phase1_pseudo`` : 3-tuple  pseudo-ternary coords of the liquid-rich phase
        - ``phase2_pseudo`` : 3-tuple  pseudo-ternary coords of the second liquid phase
        - ``phase1_4comp``  : np.ndarray  4-component mole fractions of phase 1
        - ``phase2_4comp``  : np.ndarray  4-component mole fractions of phase 2
    """
    results = []
    n = n_points

    for i in range(0, n + 1):  # solute  (0 → include solvent/diluent binary)
        for j in range(
            0, n - i + 1
        ):  # pseudo-solvent (0 → include solute/diluent binary)
            k = n - i - j  # diluent (0 → include solute/solvent binary)
            # Skip pure-component corners (at least two axes zero simultaneously)
            if (i == 0) + (j == 0) + (k == 0) >= 2:
                continue

            # 4-component feed: preserve solvent sub-ratio
            x_solute = i / n
            x_solvent_total = j / n
            x_solvent1 = solvent_ratio * x_solvent_total
            x_solvent2 = (1.0 - solvent_ratio) * x_solvent_total
            x_diluent = k / n

            feed = np.array([x_solute, x_solvent1, x_solvent2, x_diluent])
            # Numerical guard: ensure feed sums to 1
            feed = feed / feed.sum()

            try:
                result = PhaseEquilibrium.tp_flash(
                    eos, T, P, feed * si.MOL, max_iter=10000
                )
            except Exception:
                continue

            # Extract compositions from both phases
            x1 = np.array(result.liquid.molefracs)
            x2 = np.array(result.vapor.molefracs)

            # Accept as LLE only if both phases are liquid.
            # density is molar (mol/m³); gas at 1 atm/25°C ≈ 41 mol/m³, liquids > 1000 mol/m³.
            _mol_m3 = si.MOL / si.METER**3
            rho1 = result.liquid.density / _mol_m3
            rho2 = result.vapor.density / _mol_m3
            if rho1 <= 600 or rho2 <= 600:
                continue

            # Reject trivial (identical) phases
            if np.allclose(x1, x2, atol=1e-4):
                continue

            results.append(
                {
                    "feed_pseudo":   _to_pseudo_ternary(feed),
                    "phase1_pseudo": _to_pseudo_ternary(x1),
                    "phase2_pseudo": _to_pseudo_ternary(x2),
                    "feed_4comp":    feed,
                    "phase1_4comp":  x1,
                    "phase2_4comp":  x2,
                }
            )

    return results
