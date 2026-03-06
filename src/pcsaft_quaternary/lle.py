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
        pure_records = _apply_induced_association(
            pure_records, diluent_idx, target_names
        )
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
    _mol_m3 = si.MOL / si.METER**3

    def _flash_lle(feed):
        f = np.where(feed < 1e-6, 1e-6, feed)
        f = f / f.sum()
        try:
            result = PhaseEquilibrium.tp_flash(eos, T, P, f * si.MOL, max_iter=50)
        except Exception:
            return None
        x1 = np.array(result.liquid.molefracs)
        x2 = np.array(result.vapor.molefracs)
        rho1 = result.liquid.density / _mol_m3
        rho2 = result.vapor.density / _mol_m3
        if rho1 <= 600 or rho2 <= 600:
            return None
        if np.allclose(x1, x2, atol=1e-4):
            return None
        return f, x1, x2

    results = []
    n = n_points

    # --- interior scan ---
    for i in range(1, n + 1):
        for j in range(1, n - i + 1):
            k = n - i - j
            if k < 1:
                continue
            feed = np.array([i / n, j / n, k / n], dtype=float)
            hit = _flash_lle(feed)
            if hit is None:
                continue
            f, x1, x2 = hit
            results.append({
                "feed_pseudo":   tuple(float(v) for v in f),
                "phase1_pseudo": tuple(float(v) for v in x1),
                "phase2_pseudo": tuple(float(v) for v in x2),
                "feed_3comp":    f,
                "phase1_3comp":  x1,
                "phase2_3comp":  x2,
            })

    # --- binary edge scan: one representative per edge ---
    edges = [
        [np.array([0.0, j/n, (n-j)/n]) for j in range(1, n)],  # A=0
        [np.array([i/n, 0.0, (n-i)/n]) for i in range(1, n)],  # B=0
        [np.array([i/n, (n-i)/n, 0.0]) for i in range(1, n)],  # C=0
    ]
    for edge in edges:
        for feed in edge:
            hit = _flash_lle(feed)
            if hit is None:
                continue
            f, x1, x2 = hit
            results.append({
                "feed_pseudo":   tuple(float(v) for v in f),
                "phase1_pseudo": tuple(float(v) for v in x1),
                "phase2_pseudo": tuple(float(v) for v in x2),
                "feed_3comp":    f,
                "phase1_3comp":  x1,
                "phase2_3comp":  x2,
            })
            break

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
    _mol_m3 = si.MOL / si.METER**3

    def _flash_lle(feed):
        """Return (x1, x2) if feed splits into two liquids, else None."""
        f = np.where(feed < 1e-6, 1e-6, feed)
        f = f / f.sum()
        try:
            result = PhaseEquilibrium.tp_flash(eos, T, P, f * si.MOL, max_iter=50)
        except Exception:
            return None
        x1 = np.array(result.liquid.molefracs)
        x2 = np.array(result.vapor.molefracs)
        rho1 = result.liquid.density / _mol_m3
        rho2 = result.vapor.density / _mol_m3
        if rho1 <= 600 or rho2 <= 600:
            return None
        if np.allclose(x1, x2, atol=1e-4):
            return None
        return f, x1, x2

    results = []
    n = n_points

    # --- interior scan ---
    for i in range(1, n + 1):
        for j in range(1, n - i + 1):
            k = n - i - j
            if k < 1:
                continue
            x_solvent_total = j / n
            feed = np.array([
                i / n,
                solvent_ratio * x_solvent_total,
                (1.0 - solvent_ratio) * x_solvent_total,
                k / n,
            ])
            hit = _flash_lle(feed)
            if hit is None:
                continue
            f, x1, x2 = hit
            results.append({
                "feed_pseudo":   _to_pseudo_ternary(f),
                "phase1_pseudo": _to_pseudo_ternary(x1),
                "phase2_pseudo": _to_pseudo_ternary(x2),
                "feed_4comp":    f,
                "phase1_4comp":  x1,
                "phase2_4comp":  x2,
            })

    # --- binary edge scan: one representative per edge ---
    # Edge 1: solute = 0  (pseudo-solvent / diluent binary)
    # Edge 2: pseudo-solvent = 0  (solute / diluent binary)
    # Edge 3: diluent = 0  (solute / pseudo-solvent binary)
    edges = [
        [np.array([0.0, solvent_ratio * (j/n), (1-solvent_ratio) * (j/n), (n-j)/n])
         for j in range(1, n)],
        [np.array([i/n, 0.0, 0.0, (n-i)/n])
         for i in range(1, n)],
        [np.array([i/n, solvent_ratio * ((n-i)/n), (1-solvent_ratio) * ((n-i)/n), 0.0])
         for i in range(1, n)],
    ]
    for edge in edges:
        for feed in edge:
            hit = _flash_lle(feed)
            if hit is None:
                continue
            f, x1, x2 = hit
            results.append({
                "feed_pseudo":   _to_pseudo_ternary(f),
                "phase1_pseudo": _to_pseudo_ternary(x1),
                "phase2_pseudo": _to_pseudo_ternary(x2),
                "feed_4comp":    f,
                "phase1_4comp":  x1,
                "phase2_4comp":  x2,
            })
            break  # one per edge is enough

    return results


def suggest_experiments(
    tie_line_data,
    eos,
    T,
    P,
    n=5,
    target_phi=0.5,
    mass_basis=True,
    molar_masses=None,
    output=None,
    names_pseudo=None,
):
    """Return n feed compositions that are well-spread across the two-phase
    envelope and yield roughly equal phase volumes.

    Parameters
    ----------
    tie_line_data : list[dict]
        Output of ``pseudoternary_lle`` / ``ternary_lle`` (the raw scan data,
        which contains ``phase1_4comp`` / ``phase2_4comp`` keys for pseudoternary
        systems, or ``phase1_3comp`` / ``phase2_3comp`` for true ternary systems).
    eos : feos.EquationOfState
        PC-SAFT EOS used to re-flash candidate feed compositions.
    T : si_units.SIObject
        Temperature (same value used for the original scan).
    P : si_units.SIObject
        Pressure (same value used for the original scan).
    n : int
        Number of experiment points to return (default 5).
    target_phi : float
        Target volumetric fraction of phase 1 (default 0.5).
    mass_basis : bool
        If True (default), return ``z_pseudo`` in mass fractions.
        Requires ``molar_masses``.
    molar_masses : np.ndarray or None
        Molar masses in g/mol (required when ``mass_basis=True``).
    output : str or None
        If given, save a ternary diagram with the suggested points overlaid to
        this path.  Requires ``names_pseudo``.
    names_pseudo : list[str] or None
        Axis labels ``[solute, pseudo_solvent, diluent]`` used when ``output``
        is set.

    Returns
    -------
    list[dict]
        One dict per selected composition:

        - ``z_pseudo``     : 3-tuple  pseudo-ternary coords (mass or mole frac)
        - ``z_4comp``      : np.ndarray  feed mole fractions (4- or 3-component)
        - ``phi1``         : float  actual volumetric fraction of phase 1
        - ``alpha``        : float  position on tie-line (0 = phase2, 1 = phase1)
        - ``phase1_4comp`` : np.ndarray  equilibrium phase 1 mole fractions
        - ``phase2_4comp`` : np.ndarray  equilibrium phase 2 mole fractions
    """
    if not tie_line_data:
        return []

    # Detect pseudoternary vs true ternary
    is_pseudo = "phase1_4comp" in tie_line_data[0]
    p1_key = "phase1_4comp" if is_pseudo else "phase1_3comp"
    p2_key = "phase2_4comp" if is_pseudo else "phase2_3comp"

    # Drop binary-edge tie-lines: both phase endpoints near zero in the same
    # pseudo-ternary component → the tie-line sits on an edge of the diagram.
    _EDGE_TOL = 1e-2
    interior_data = [
        tl for tl in tie_line_data
        if not any(
            tl["phase1_pseudo"][k] < _EDGE_TOL and tl["phase2_pseudo"][k] < _EDGE_TOL
            for k in range(3)
        )
    ]
    if not interior_data:
        interior_data = tie_line_data  # fallback: use all if none survive

    _mol_m3 = si.MOL / si.METER**3

    def _phase_volumes(p1, p2):
        """Flash at the tie-line midpoint; return (v1, v2) as plain floats [m³]
        read directly from the liquid/vapor State objects, or None on failure.

        Because both phases carry 0.5 mol at the midpoint, v1 and v2 are
        proportional to the molar volumes — their ratio is all that matters for
        the equal-volume formula.
        """
        mid = 0.5 * p1 + 0.5 * p2
        f = np.where(mid < 1e-9, 1e-9, mid)
        f = f / f.sum()
        try:
            res = PhaseEquilibrium.tp_flash(eos, T, P, f * si.MOL, max_iter=100)
        except Exception:
            return None
        # Liquid-phase guard via molar density (established threshold: > 600 mol/m³)
        rho1 = float(res.liquid.density / _mol_m3)
        rho2 = float(res.vapor.density / _mol_m3)
        if rho1 <= 600 or rho2 <= 600:
            return None
        if np.allclose(
            np.array(res.liquid.molefracs), np.array(res.vapor.molefracs), atol=1e-4
        ):
            return None
        v1 = float(res.liquid.volume / si.METER**3)
        v2 = float(res.vapor.volume / si.METER**3)
        return v1, v2

    # Minimum pseudo-ternary component in the feed to be considered "well inside"
    _INTERIOR_TOL = 0.02

    # Step 1: for each tie-line get phase volumes (one flash), then solve for the
    # equal-volume alpha analytically.
    #
    # Because z = α·p1 + (1−α)·p2, the lever rule gives β = α exactly, so:
    #   φ1(α) = (α·v1) / (α·v1 + (1−α)·v2)
    # Setting φ1 = t and solving:
    #   α = (t·v2) / ((1−t)·v1 + t·v2)
    # v1, v2 need only be proportional to molar volumes (same moles at midpoint).
    candidates = []
    for tl in interior_data:
        p1 = np.array(tl[p1_key])
        p2 = np.array(tl[p2_key])
        if np.allclose(p1, p2, atol=1e-4):
            continue
        vols = _phase_volumes(p1, p2)
        if vols is None:
            continue
        v1, v2 = vols
        t = target_phi
        alpha = float(np.clip(
            (t * v2) / ((1.0 - t) * v1 + t * v2),
            0.01, 0.99,
        ))
        phi1 = (alpha * v1) / (alpha * v1 + (1.0 - alpha) * v2)
        z = alpha * p1 + (1.0 - alpha) * p2

        if is_pseudo:
            if mass_basis and molar_masses is not None:
                z_pseudo = _to_pseudo_ternary_mass(z, molar_masses)
            else:
                z_pseudo = _to_pseudo_ternary(z)
        else:
            if mass_basis and molar_masses is not None:
                z_pseudo = _to_ternary_mass(z, molar_masses)
            else:
                z_pseudo = tuple(float(v) for v in z)

        # Skip feeds where any pseudo-ternary component is near zero
        if any(v < _INTERIOR_TOL for v in z_pseudo):
            continue

        tl_len = float(np.linalg.norm(p1 - p2))
        candidates.append({
            "z_pseudo": z_pseudo,
            "z_4comp": z,
            "phi1": phi1,
            "alpha": alpha,
            "phase1_4comp": p1,
            "phase2_4comp": p2,
            "_tl_len": tl_len,
        })

    if not candidates:
        return []

    n_select = min(n, len(candidates))

    # Step 2: greedy farthest-point (maximin) sampling
    seed_idx = max(range(len(candidates)), key=lambda i: candidates[i]["_tl_len"])
    selected_indices = [seed_idx]
    remaining_indices = [i for i in range(len(candidates)) if i != seed_idx]

    while len(selected_indices) < n_select and remaining_indices:
        sel_z = [candidates[i]["z_4comp"] for i in selected_indices]

        def _min_dist(i):
            z_i = candidates[i]["z_4comp"]
            return min(float(np.linalg.norm(z_i - s)) for s in sel_z)

        next_idx = max(remaining_indices, key=_min_dist)
        selected_indices.append(next_idx)
        remaining_indices.remove(next_idx)

    # Build output, removing internal keys
    result = []
    for i in selected_indices:
        c = {k: v for k, v in candidates[i].items() if not k.startswith("_")}
        result.append(c)

    if output is not None and names_pseudo is not None and tie_line_data:
        from .plot import plot_pseudoternary_lle
        T_K = float(T / si.KELVIN)
        P_Pa = float(P / si.PASCAL)
        from pathlib import Path as _Path
        out_path = str(_Path(output).with_suffix(".pdf")) if not _Path(output).suffix else output
        plot_pseudoternary_lle(
            tie_line_data, names_pseudo, T_K, P_Pa, out_path,
            suggested_points=result,
        )

    return result
