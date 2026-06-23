"""
r255x_validation_visuals.py

Bar plots for R255X GAT model predictions against verified percent editing
in trans, styled to match plot_results.py.

Outputs:
    figures/r255x_gat_intrans_regular.png
    figures/r255x_gat_intrans_bars_80pct_transparent.png
    figures/r255x_gat_model_bars_80pct_transparent.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# ── Aesthetic setup ───────────────────────────────────────────────────────────

PALETTE = dict(
    in_trans="#2E6A9E",   # deep steel blue
    model="#C8952A",      # dark gold
    bg="#FAFAFA",
    grid="#EBEBEB",
    text="#1C2B3A",
    subtext="#637080",
)


def setup_style():
    mpl.rcParams.update({
        "figure.facecolor":   PALETTE["bg"],
        "axes.facecolor":     "#FFFFFF",
        "axes.edgecolor":     "#CCCCCC",
        "axes.linewidth":     0.8,
        "axes.grid":          True,
        "grid.color":         PALETTE["grid"],
        "grid.linewidth":     0.7,
        "grid.linestyle":     "-",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "xtick.color":        PALETTE["subtext"],
        "ytick.color":        PALETTE["subtext"],
        "xtick.labelsize":    8.5,
        "ytick.labelsize":    8.5,
        "axes.labelsize":     9.5,
        "axes.labelcolor":    PALETTE["text"],
        "font.family":        "sans-serif",
        "font.sans-serif":    ["IBM Plex Sans", "Helvetica Neue", "Arial"],
        "legend.framealpha":  0.92,
        "legend.edgecolor":   "#CCCCCC",
        "legend.fontsize":    8.5,
        "figure.dpi":         100,
    })


# ── Data ─────────────────────────────────────────────────────────────────────

r255x_preds = {
    "GGAC": 0.60976243,
    "AC":   0.20690373,
    "ND1":  0.0939541,
    "ND2":  0.07556008,
    "ND3":  0.16226596,
    "ND4":  0.13881683,
    "ND5":  0.0924708,
    "ND6":  0.13266876,
    "ND7":  0.11599071,
    "ND9":  0.08470311,
    "ND10": 0.13723879,
    "ND11": 0.11418517,
    "ND12": 0.08059392,
    "ND13": 0.386976,
    "ND14": 0.12168525,
}

r255x_val = pd.DataFrame({
    "Guide": ["GGAC", "AC", "ND1", "ND2", "ND3", "ND4", "ND5", "ND6",
              "ND7", "ND9", "ND10", "ND11", "ND12", "ND13", "ND14"],
    "Rep1": [63.4, 20.9, 0.2, 3.33, 8.71, 5.92, 4.79, 14.6, 12.9, 35.1, 6.0, 9.6, 26.9, 10.8, 41.7],
    "Rep2": [58.9, 13.0, 0.8, 4.78, 9.41, 5.48, 6.80, 9.88, 28.3, 48.7, 8.9, 13.1, 28.0, 12.3, 23.5],
    "Rep3": [47.4, 7.82, 2.4, 0.69, 10.1, 7.84, np.nan, 1.54, 10.3, 48.9, 46.5, 7.0, 30.5, 15.3, 15.0],
    "Rep4": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 16.2, np.nan, np.nan, np.nan, 1.9],
    "Rep5": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 10.9, np.nan, np.nan, np.nan, 5.2],
    "Rep6": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 6.2, np.nan, np.nan, np.nan, np.nan],
})


# ── Build plotting table ─────────────────────────────────────────────────────

def build_plot_df():
    rep_cols = [c for c in r255x_val.columns if c.startswith("Rep")]

    df = r255x_val.copy()
    df["true"] = df[rep_cols].mean(axis=1, skipna=True) / 100
    df["sd"] = df[rep_cols].std(axis=1, skipna=True) / 100
    df["pred"] = df["Guide"].map(r255x_preds)

    df["true_min"] = (df["true"] - df["sd"]).clip(lower=0)
    df["true_max"] = (df["true"] + df["sd"]).clip(upper=1)

    return df


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_r255x_gat_bars(
    df,
    out_path,
    alpha_in_trans=0.85,
    alpha_model=0.85,
    dpi=200,
    show=True,
):
    setup_style()

    guides = df["Guide"].to_numpy()
    x = np.arange(len(guides))
    width = 0.34

    true = df["true"].to_numpy()
    pred = df["pred"].to_numpy()

    yerr_lower = true - df["true_min"].to_numpy()
    yerr_upper = df["true_max"].to_numpy() - true

    fig = plt.figure(figsize=(10.8, 6.6), facecolor=PALETTE["bg"])

    # Main axes; header band from y=0.88 to y=1.00 reserved for title + legend.
    ax = fig.add_axes([0.08, 0.15, 0.90, 0.70])

    bars_true = ax.bar(
        x - width / 2,
        true,
        width,
        color=PALETTE["in_trans"],
        alpha=alpha_in_trans,
        label="Reaction in trans (validation)",
        edgecolor="white",
        linewidth=0.35,
        zorder=3,
    )

    bars_pred = ax.bar(
        x + width / 2,
        pred,
        width,
        color=PALETTE["model"],
        alpha=alpha_model,
        label="Model prediction",
        edgecolor="white",
        linewidth=0.35,
        zorder=3,
    )

    ax.errorbar(
        x - width / 2,
        true,
        yerr=np.vstack([yerr_lower, yerr_upper]),
        fmt="none",
        ecolor=PALETTE["in_trans"],
        elinewidth=1.0,
        capsize=3,
        capthick=1.0,
        alpha=alpha_in_trans,
        zorder=4,
    )

    ax.set_xlabel("gRNA ID")
    ax.set_ylabel("Editing rate (Sanger seq.)")

    ax.set_xticks(x)
    ax.set_xticklabels(guides, rotation=45, ha="right")

    ymax = max(df["true_max"].max(), df["pred"].max()) * 1.12
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1.0, decimals=0))

    # Title
    fig.text(
        0.08, 0.965,
        "R255X GAT Model Predictions Against Verified Percent Editing In Trans",
        ha="left", va="top",
        fontsize=13, fontweight="semibold",
        color=PALETTE["text"],
    )

    # Legend sits just below the title, flush left
    fig.legend(
        handles=[bars_true, bars_pred],
        labels=["Reaction in trans (validation)", "Model prediction"],
        loc="upper left",
        bbox_to_anchor=(0.08, 0.918),
        ncol=2,
        frameon=True,
        columnspacing=1.2,
        handlelength=1.6,
        handletextpad=0.5,
    )

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=PALETTE["bg"])
    print(f"saved → {out_path}")

    if show:
        plt.show()

    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

r255x_mle_static = {
    "ND1":  90.9,
    "ND2":  86.7,
    "ND3":  86.7,
    "ND4":  83.7,
    "ND5":  81.8,
    "ND6":  80.6,
    "ND7":  75.5,
    "ND8":  66.7,
    "ND9":  69.2,
    "ND10": 70.6,
    "ND11": 70.6,
    "ND12": 71.8,
    "ND13": 68.2,
    "ND14": 70.0,
}


def load_mle_data(
    csv_path="r255x_natalie.csv",
    json_path="../data/ref/guides.json",
    z=1.96,
):
    """
    Load binomial (n, k) data from csv_path, keyed by guide ID via guides.json.

    guides.json is expected to be one of:
      • {"ND1": "ACGT...", "ND2": ...}            flat: value is the 5to3 seq
      • {"ND1": {"5to3": "ACGT...", ...}, ...}    nested: value is a dict with a '5to3' key

    Returns a dict: {"ND1": {"mle": float, "lower": float, "upper": float}, ...}
    """
    import json

    with open(json_path) as f:
        guides_raw = json.load(f)

    # guides.json is a list of {guide_id, seq, target, ...}
    # guide_id is lowercase (e.g. "nd1"); seq is RNA (U not T)
    guide_to_seq = {
        entry["guide_id"].upper(): entry["seq"].strip().upper().replace("U", "T")
        for entry in guides_raw
        if "guide_id" in entry and "seq" in entry
    }

    df = pd.read_csv(csv_path)
    df["5to3"] = df["5to3"].str.strip().str.upper()

    # Keep only the ND1-ND14 guides present in guides.json
    target_ids = [gid for gid in guide_to_seq if gid in r255x_mle_static]

    results = {}
    missing = []
    for gid in target_ids:
        seq = guide_to_seq[gid]
        row = df[df["5to3"] == seq]
        if row.empty:
            missing.append(gid)
            continue
        n = int(row["n"].iloc[0])
        k = int(row["k"].iloc[0])
        p = k / n
        margin = z * np.sqrt(p * (1 - p) / n) if n > 0 else 0.0
        results[gid] = dict(
            mle=p,
            lower=max(0.0, p - margin),
            upper=min(1.0, p + margin),
        )

    if missing:
        print(f"[load_mle_data] warning: no CSV row found for {missing}")

    return results


def plot_mle_bars(out_path, mle_data=None, dpi=200, show=True):
    """
    mle_data: output of load_mle_data(). If None, falls back to static point
              estimates with no error bars.
    """
    setup_style()

    if mle_data is not None:
        guides = [g for g in r255x_mle_static if g in mle_data]
        values = np.array([mle_data[g]["mle"]   for g in guides])
        lowers = np.array([mle_data[g]["lower"]  for g in guides])
        uppers = np.array([mle_data[g]["upper"]  for g in guides])
        yerr   = np.vstack([values - lowers, uppers - values])
    else:
        guides = list(r255x_mle_static.keys())
        values = np.array(list(r255x_mle_static.values())) / 100
        yerr   = None

    x = np.arange(len(guides))

    fig = plt.figure(figsize=(10.8, 6.6), facecolor=PALETTE["bg"])
    ax = fig.add_axes([0.08, 0.15, 0.90, 0.70])

    ax.bar(
        x, values,
        width=0.55,
        color=PALETTE["in_trans"],
        alpha=0.85,
        edgecolor="white",
        linewidth=0.35,
        zorder=3,
    )

    if yerr is not None:
        ax.errorbar(
            x, values,
            yerr=yerr,
            fmt="none",
            ecolor=PALETTE["in_trans"],
            elinewidth=1.0,
            capsize=3,
            capthick=1.0,
            alpha=0.85,
            zorder=4,
        )

    ax.set_xlabel("gRNA ID")
    ax.set_ylabel("Editing rate (MLE)")
    ax.set_xticks(x)
    ax.set_xticklabels(guides, rotation=45, ha="right")
    ax.set_ylim(0, (values + (yerr[1] if yerr is not None else 0)).max() * 1.12)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1.0, decimals=0))

    fig.text(
        0.08, 0.965,
        "Percent Editing Per Top EMERGe Candidates",
        ha="left", va="top",
        fontsize=13, fontweight="semibold",
        color=PALETTE["text"],
    )

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=PALETTE["bg"])
    print(f"saved → {out_path}")

    if show:
        plt.show()

    plt.close(fig)


def main():
    os.makedirs("figures", exist_ok=True)

    plot_df = build_plot_df()

    plot_r255x_gat_bars(
        plot_df,
        "figures/r255x_gat_intrans_regular.png",
        alpha_in_trans=0.85,
        alpha_model=0.85,
    )

    plot_r255x_gat_bars(
        plot_df,
        "figures/r255x_gat_intrans_bars_80pct_transparent.png",
        alpha_in_trans=0.20,
        alpha_model=0.85,
    )

    plot_r255x_gat_bars(
        plot_df,
        "figures/r255x_gat_model_bars_80pct_transparent.png",
        alpha_in_trans=0.85,
        alpha_model=0.20,
    )

    mle_data = load_mle_data()
    plot_mle_bars("figures/r255x_emerge_mle.png", mle_data=mle_data)


if __name__ == "__main__":
    main()
