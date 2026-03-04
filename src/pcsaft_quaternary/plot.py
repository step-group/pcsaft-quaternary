"""Publication-quality pseudoternary LLE phase diagram using python-ternary."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ternary


def plot_pseudoternary_lle(tie_line_data, names_pseudo, T_K, P_Pa, output_path):
    """Generate and save a pseudoternary LLE phase diagram.

    Parameters
    ----------
    tie_line_data : list[dict]
        Output of ``scan_pseudoternary``. Each dict must have keys
        ``phase1_pseudo`` and ``phase2_pseudo`` (3-tuples: solute, pseudo-solvent, diluent).
    names_pseudo : list[str]
        Labels for the three pseudo-ternary axes: [solute_name, pseudo_solvent_label, diluent_name].
    T_K : float
        Temperature in Kelvin (used in title/print).
    P_Pa : float
        Pressure in Pascal (used in title/print).
    output_path : str
        Destination path for the PNG file (saved at 300 dpi).
    """
    if not tie_line_data:
        print("No LLE tie-lines found — diagram not generated.")
        return

    # --- font / figure setup (mirrors ternary.py template) ---
    font = {"weight": "normal", "size": 16}
    plt.rc("font", **font)

    fig, ax = plt.subplots(figsize=(3.93701, 3.1496))
    figure, tax = ternary.figure(ax=ax, scale=1)
    figure.set_size_inches(5, 5)

    # Boundary and grid
    tax.boundary(linewidth=1.0)
    tax.gridlines(color="black", multiple=0.1, linewidth=0.5)
    tax.ticks(
        axis="lbr",
        multiple=0.2,
        linewidth=1,
        tick_formats="%.1f",
        fontsize=14,
        offset=0.03,
    )

    # --- collect phase endpoints ---
    phase1_pts = [d["phase1_pseudo"] for d in tie_line_data]
    phase2_pts = [d["phase2_pseudo"] for d in tie_line_data]

    # Sort each branch by solute mole fraction (index 0) for a smooth binodal curve
    phase1_sorted = sorted(phase1_pts, key=lambda p: p[0])
    phase2_sorted = sorted(phase2_pts, key=lambda p: p[0])

    # Convert to numpy arrays for tax.plot.
    # Axis convention: bottom = diluent (2), right = solute (0), left = solvent (1).
    def _arr(pts):
        return np.array([[p[2], p[0], p[1]] for p in pts])

    _BINODAL   = "black"
    _TIELINE   = "#cccccc"  # light gray

    tax.plot(_arr(phase1_sorted), color=_BINODAL, linewidth=2.0, label="PC-SAFT binodal")
    tax.plot(_arr(phase2_sorted), color=_BINODAL, linewidth=2.0)

    # --- tie-lines: 10 evenly spaced, interior only (edge tielines lie on the
    # boundary and are already visible via the binodal curve) ---
    _EDGE_TOL = 1e-3
    interior_tl = [
        d for d in tie_line_data
        if not any(
            d["phase1_pseudo"][k] < _EDGE_TOL and d["phase2_pseudo"][k] < _EDGE_TOL
            for k in range(3)
        )
    ]
    sorted_tl = sorted(interior_tl, key=lambda d: d["phase2_pseudo"][0])
    n_show = min(7, len(sorted_tl))
    tl_indices = np.linspace(0, len(sorted_tl) - 1, n_show, dtype=int) if sorted_tl else []

    print(
        f"Tie-lines [{names_pseudo[0]} | {names_pseudo[1]} | {names_pseudo[2]}] "
        f"at T = {T_K - 273.15:.2f} °C, P = {P_Pa / 1e5:.3f} bar"
    )
    print("-" * 60)
    for idx in tl_indices:
        p1 = sorted_tl[idx]["phase1_pseudo"]
        p2 = sorted_tl[idx]["phase2_pseudo"]
        p1_plot = (p1[2], p1[0], p1[1])
        p2_plot = (p2[2], p2[0], p2[1])
        tax.line(
            p1_plot, p2_plot,
            linewidth=1.5,
            marker=None,
            color=_TIELINE,
            linestyle=":",
            zorder=5,
        )
        print(
            f"  Tie-line {idx:3d}: "
            f"phase1 = [{p1[0]:.4f}, {p1[1]:.4f}, {p1[2]:.4f}]  "
            f"phase2 = [{p2[0]:.4f}, {p2[1]:.4f}, {p2[2]:.4f}]"
        )

    # --- axis labels ---
    # Convention: bottom = diluent (water), right = solute, left = pseudo-solvent (solvents)
    tax.bottom_axis_label(names_pseudo[2], position=(0.55, -0.1, 0.5), fontsize=16)
    tax.right_axis_label(names_pseudo[0], fontsize=16, offset=0.20)
    tax.left_axis_label(names_pseudo[1], fontsize=16, offset=0.20)

    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"\nDiagram saved to: {output_path}")
