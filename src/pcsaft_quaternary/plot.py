"""Publication-quality pseudoternary LLE phase diagram using python-ternary."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import ternary


def plot_pseudoternary_lle(tie_line_data, names_pseudo, T_K, P_Pa, output_path, exp_tie_lines=None, suggested_points=None):
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
        Destination path for the PDF file.
    exp_tie_lines : list[dict] or None
        Optional experimental tie-lines to overlay. Each dict must have keys
        ``phase1_pseudo`` and ``phase2_pseudo`` (3-tuples: solute, pseudo-solvent, diluent),
        in mole fractions and using the same coordinate convention as ``tie_line_data``.
    suggested_points : list[dict] or None
        Optional output of ``suggest_experiments``.  Each dict must contain
        ``z_pseudo`` (3-tuple: solute, pseudo-solvent, diluent).  Points are
        plotted as numbered filled circles.
    """
    if not tie_line_data:
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

    # --- experimental tie-lines ---
    if exp_tie_lines:
        for d in exp_tie_lines:
            p1 = d["phase1_pseudo"]
            p2 = d["phase2_pseudo"]
            p1_plot = (p1[2], p1[0], p1[1])
            p2_plot = (p2[2], p2[0], p2[1])
            tax.line(
                p1_plot, p2_plot,
                linewidth=1.5,
                color="red",
                linestyle="-",
                marker="o",
                markersize=5,
                markerfacecolor="red",
                markeredgecolor="red",
                zorder=10,
            )

    # --- suggested experiment points ---
    _SUGGEST_COLOR = "#e67e00"  # orange
    if suggested_points:
        for k, sp in enumerate(suggested_points, 1):
            z = sp.z_pseudo   # (solute, pseudo-solvent, diluent)
            pt = (z[2], z[0], z[1])  # → (diluent, solute, pseudo-solvent) for tax
            tax.plot([pt], marker="o", markersize=8, color=_SUGGEST_COLOR,
                     markeredgecolor="black", markeredgewidth=0.6,
                     linestyle="", zorder=15)
            # Number label offset slightly toward the interior
            tax.annotate(
                str(k), pt,
                fontsize=10, fontweight="bold", color=_SUGGEST_COLOR,
                ha="center", va="bottom",
                xytext=(0, 7), textcoords="offset points",
                zorder=16,
            )

    # --- legend ---
    handles = [
        Line2D([0], [0], color=_BINODAL, linewidth=2, label="PC-SAFT"),
    ]
    if exp_tie_lines:
        handles.append(
            Line2D([0], [0], color="red", linewidth=1.5, marker="o", markersize=5, label="Experimental")
        )
    if suggested_points:
        handles.append(
            Line2D([0], [0], color=_SUGGEST_COLOR, marker="o", markersize=6,
                   linewidth=0, markeredgecolor="black", markeredgewidth=0.6,
                   label="Suggested")
        )
    ax.legend(handles=handles, loc="upper right", fontsize=11, framealpha=0.7)

    # --- axis labels ---
    # Convention: bottom = diluent (water), right = solute, left = pseudo-solvent (solvents)
    tax.bottom_axis_label(names_pseudo[2], position=(0.55, -0.1, 0.5), fontsize=16)
    tax.right_axis_label(names_pseudo[0], fontsize=16, offset=0.20)
    tax.left_axis_label(names_pseudo[1], fontsize=16, offset=0.20)

    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Diagram saved to: {output_path}")
