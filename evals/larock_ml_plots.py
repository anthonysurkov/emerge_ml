"""
plot_results.py — pretty static figures for ViennaLandscape GNN evals.

Usage:
    python plot_results.py --binary  evals/r255x_eval.csv
    python plot_results.py --regression evals/r270x_z_eval.csv
    python plot_results.py --binary evals/r255x_eval.csv --regression evals/r270x_z_eval.csv

    --out-binary   path to save binary figure   (default: <csv stem>_fig.png)
    --out-reg      path to save regression figure
    --dpi          output DPI (default: 200)
    --no-show      skip plt.show()
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    roc_curve, precision_recall_curve,
    roc_auc_score, accuracy_score, f1_score,
)

# ── Aesthetic setup ───────────────────────────────────────────────────────────

PALETTE = dict(
    neg         = "#2E6A9E",   # deep steel blue   (negative class)
    pos         = "#C0392B",   # red                (positive class)
    curve_roc   = "#2E6A9E",   # deep blue
    curve_pr    = "#C8952A",   # gold
    curve_f1    = "#2E6A9E",   # deep blue
    thresh_best = "#B8550A",   # burnt amber
    reg_true    = "#2E6A9E",   # deep blue
    reg_pred    = "#C8952A",   # dark gold
    resid_hist  = "#C8952A",   # gold
    diagonal    = "#999999",
    bg          = "#FAFAFA",
    grid        = "#EBEBEB",
    text        = "#1C2B3A",
    subtext     = "#637080",
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
        "axes.titlesize":     10.5,
        "axes.titleweight":   "semibold",
        "axes.titlecolor":    PALETTE["text"],
        "axes.titlelocation": "left",
        "font.family":        "sans-serif",
        "font.sans-serif":    ["IBM Plex Sans", "Helvetica Neue", "Arial"],
        "legend.framealpha":  0.9,
        "legend.edgecolor":   "#CCCCCC",
        "legend.fontsize":    8.5,
        "figure.dpi":         100,
    })

def stat_box(ax, text, loc="upper right", fontsize=8):
    props = dict(boxstyle="round,pad=0.45", facecolor="white",
                 edgecolor="#CCCCCC", alpha=0.88)
    xmap = {"upper right": 0.97, "upper left": 0.03,
            "lower right": 0.97, "lower left": 0.03}
    ymap = {"upper right": 0.97, "upper left": 0.97,
            "lower right": 0.05, "lower left": 0.05}
    ha = "right" if "right" in loc else "left"
    va = "top"   if "upper" in loc else "bottom"
    ax.text(xmap[loc], ymap[loc], text, transform=ax.transAxes,
            fontsize=fontsize, va=va, ha=ha,
            color=PALETTE["text"], bbox=props, linespacing=1.55,
            fontfamily="monospace")

def panel_label(ax, letter):
    ax.text(-0.12, 1.07, letter, transform=ax.transAxes,
            fontsize=13, fontweight="bold", color=PALETTE["text"], va="top")


# ── Best-F1 threshold ─────────────────────────────────────────────────────────

def best_f1_threshold(true, pred_p):
    prec, rec, thr = precision_recall_curve(true, pred_p)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    i = f1s[:-1].argmax()
    return thr[i], f1s[i], prec[i], rec[i]


# ── Binary figure ─────────────────────────────────────────────────────────────

def plot_binary(df, out_path, dpi=200, show=True):
    setup_style()

    true   = df["true"].values.astype(int)
    pred_p = df["pred"].values
    idx    = df["idx"].values

    best_t = float(df["best_threshold"].iloc[0]) if "best_threshold" in df.columns \
             else best_f1_threshold(true, pred_p)[0]

    pos_rate                          = true.mean()
    auc                               = roc_auc_score(true, pred_p)
    f1_05                             = f1_score(true, (pred_p >= 0.5).astype(int))
    acc_05                            = accuracy_score(true, (pred_p >= 0.5).astype(int))
    best_t, best_f1, best_prec, best_rec = best_f1_threshold(true, pred_p)
    acc_best                          = accuracy_score(true, (pred_p >= best_t).astype(int))

    fpr, tpr, _          = roc_curve(true, pred_p)
    prec_c, rec_c, thr_c = precision_recall_curve(true, pred_p)
    f1_curve             = 2 * prec_c * rec_c / (prec_c + rec_c + 1e-9)

    fig = plt.figure(figsize=(13, 9), facecolor=PALETTE["bg"])
    fig.suptitle("Binary Classification Evaluation",
                 fontsize=15, fontweight="semibold", color=PALETTE["text"],
                 x=0.5, y=1.01, ha="center")
    gs   = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]

    # A: Scatter coloured by true label
    ax = axes[0];  panel_label(ax, "A")
    ax.scatter(idx[true == 0], pred_p[true == 0],
               c=PALETTE["neg"], s=5, alpha=0.32, linewidths=0,
               label="Negative", rasterized=True, zorder=2)
    ax.scatter(idx[true == 1], pred_p[true == 1],
               c=PALETTE["pos"], s=5, alpha=0.70, linewidths=0,
               label="Positive", rasterized=True, zorder=3)
    ax.axhline(best_t, color=PALETTE["thresh_best"], lw=0.9, ls="--",
               label=f"Threshold = {best_t:.3f}")
    ax.set_xlabel("Sample index");  ax.set_ylabel("Predicted probability")
    ax.set_title("Predictions by Sample");  ax.set_ylim(-0.04, 1.04)
    ax.legend(loc="center right", markerscale=1.8)

    # B: ROC curve
    ax = axes[1];  panel_label(ax, "B")
    ax.fill_between(fpr, tpr, alpha=0.10, color=PALETTE["curve_roc"])
    ax.plot(fpr, tpr, color=PALETTE["curve_roc"], lw=2.2)
    ax.plot([0, 1], [0, 1], color=PALETTE["diagonal"], lw=1, ls=":")
    ax.set_xlabel("False positive rate");  ax.set_ylabel("True positive rate")
    ax.set_title("ROC Curve");  ax.set_xlim(0, 1);  ax.set_ylim(0, 1)
    stat_box(ax, f"AUC = {auc:.3f}", loc="lower right")

    # C: Precision-Recall curve
    ax = axes[2];  panel_label(ax, "C")
    ax.fill_between(rec_c, prec_c, alpha=0.10, color=PALETTE["curve_pr"])
    ax.plot(rec_c, prec_c, color=PALETTE["curve_pr"], lw=2.2)
    ax.axhline(pos_rate, color=PALETTE["diagonal"], lw=1, ls=":",
               label=f"Baseline ({pos_rate:.1%})")
    ax.scatter([best_rec], [best_prec], color=PALETTE["thresh_best"],
               s=55, zorder=5, marker="D", edgecolors="white", linewidths=0.6,
               label=f"Best F1 = {best_f1:.3f}")
    ax.set_xlabel("Recall");  ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve");  ax.set_xlim(0, 1);  ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    stat_box(ax,
             f"Best F1    {best_f1:.3f}\n"
             f"Precision  {best_prec:.3f}\n"
             f"Recall     {best_rec:.3f}\n"
             f"Threshold  {best_t:.3f}",
             loc="lower left")

    # D: F1 vs threshold
    ax = axes[3];  panel_label(ax, "D")
    ax.plot(thr_c, f1_curve[:-1], color=PALETTE["curve_f1"], lw=2.2)
    ax.axvline(best_t, color=PALETTE["thresh_best"], lw=1.4, ls="--",
               label=f"Threshold {best_t:.3f} → F1 = {best_f1:.3f}")
    ax.set_xlabel("Threshold");  ax.set_ylabel("F1 score")
    ax.set_title("F1 vs. Decision Threshold");  ax.set_xlim(0, 1)
    ax.legend(loc="upper right")
    stat_box(ax,
             f"@ {best_t:.3f}  acc {acc_best:.3f}  F1 {best_f1:.3f}",
             loc="lower left")

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=PALETTE["bg"])
    print(f"[binary] saved → {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_scatter_highlight(df, out_path, highlight="pos", dpi=200, show=True):
    """
    Standalone predictions-by-sample plot with one class dimmed ~80%.
      highlight="pos"  → positives vivid, negatives ghosted
      highlight="neg"  → negatives vivid, positives ghosted
    """
    setup_style()

    true   = df["true"].values.astype(int)
    pred_p = df["pred"].values
    idx    = df["idx"].values

    best_t = float(df["best_threshold"].iloc[0]) if "best_threshold" in df.columns \
             else best_f1_threshold(true, pred_p)[0]

    pos_rate = true.mean()
    auc      = roc_auc_score(true, pred_p)

    alpha_neg = 0.10 if highlight == "pos" else 0.32
    alpha_pos = 0.22 if highlight == "neg" else 0.70

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=PALETTE["bg"])
    ax.set_facecolor("#FFFFFF")

    ax.scatter(idx[true == 0], pred_p[true == 0],
               c=PALETTE["neg"], s=5, alpha=alpha_neg, linewidths=0,
               label="Negative", rasterized=True, zorder=2)
    ax.scatter(idx[true == 1], pred_p[true == 1],
               c=PALETTE["pos"], s=5, alpha=alpha_pos, linewidths=0,
               label="Positive", rasterized=True, zorder=3)
    ax.axhline(best_t, color=PALETTE["thresh_best"], lw=0.9, ls="--",
               label=f"Threshold = {best_t:.3f}")

    ax.set_xlabel("Sample index")
    ax.set_ylabel("Predicted probability")
    ax.set_ylim(-0.04, 1.04)
    ax.set_title("Predictions by Sample", fontweight="semibold",
                 color=PALETTE["text"], loc="left")
    ax.legend(loc="center right", markerscale=1.8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=PALETTE["bg"])
    print(f"[scatter:{highlight}] saved → {out_path}")
    if show:
        plt.show()
    plt.close(fig)


# ── Regression figure ─────────────────────────────────────────────────────────

def plot_regression(df, out_path, dpi=200, show=True):
    setup_style()

    true  = df["true"].values
    pred  = df["pred"].values
    idx   = df["idx"].values
    resid = pred - true

    ss_res = (resid ** 2).sum()
    ss_tot = ((true - true.mean()) ** 2).sum()
    r2   = 1.0 - ss_res / ss_tot
    rmse = np.sqrt((resid ** 2).mean())
    mae  = np.abs(resid).mean()

    mn  = min(true.min(), pred.min())
    mx  = max(true.max(), pred.max())
    pad = (mx - mn) * 0.05

    fig = plt.figure(figsize=(13, 9), facecolor=PALETTE["bg"])
    fig.suptitle("Regression Evaluation",
                 fontsize=15, fontweight="semibold", color=PALETTE["text"],
                 x=0.5, y=1.01, ha="center")
    gs   = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]

    # A: True vs predicted scatter, coloured by |residual|
    ax = axes[0];  panel_label(ax, "A")
    abs_resid = np.abs(resid)
    sc = ax.scatter(true, pred, c=abs_resid, cmap="YlOrRd",
                    s=6, alpha=0.60, linewidths=0, rasterized=True, zorder=2)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("|residual|", fontsize=8, color=PALETTE["subtext"])
    cbar.ax.tick_params(labelsize=7.5)
    ax.plot([mn-pad, mx+pad], [mn-pad, mx+pad],
            color=PALETTE["diagonal"], lw=1.3, ls="--", zorder=3)
    ax.set_xlim(mn-pad, mx+pad);  ax.set_ylim(mn-pad, mx+pad)
    ax.set_xlabel("True value");   ax.set_ylabel("Predicted value")
    ax.set_title("Predicted vs. True")
    stat_box(ax, f"R²    {r2:.4f}\nRMSE  {rmse:.4f}\nMAE   {mae:.4f}", loc="upper left")

    # B: True & predicted over index
    ax = axes[1];  panel_label(ax, "B")
    order = np.argsort(idx)
    ax.plot(idx[order], true[order],
            color=PALETTE["reg_true"], lw=1.3, alpha=0.85, label="True")
    ax.plot(idx[order], pred[order],
            color=PALETTE["reg_pred"], lw=1.3, alpha=0.85, label="Predicted")
    ax.set_xlabel("Sample index");  ax.set_ylabel("Value")
    ax.set_title("True & Predicted over Index")
    ax.legend(loc="upper right")

    # C: Residual distribution with KDE
    ax = axes[2];  panel_label(ax, "C")
    sns.histplot(resid, bins=50, ax=ax,
                 color=PALETTE["resid_hist"], alpha=0.72,
                 edgecolor="white", linewidth=0.3,
                 kde=True, kde_kws=dict(lw=1.8, color=PALETTE["text"]))
    ax.axvline(0,            color=PALETTE["diagonal"], lw=1.2, ls="--")
    ax.axvline(resid.mean(), color=PALETTE["reg_pred"], lw=1.2, ls=":",
               label=f"mean = {resid.mean():.4f}")
    ax.set_xlabel("Residual (pred − true)");  ax.set_ylabel("Count")
    ax.set_title("Residual Distribution")
    ax.legend(loc="upper right")
    stat_box(ax,
             f"std   {resid.std():.4f}\n"
             f"skew  {float(pd.Series(resid).skew()):.3f}",
             loc="upper left")

    # D: Residuals vs predicted (heteroscedasticity check)
    ax = axes[3];  panel_label(ax, "D")
    ax.scatter(pred, resid, c=abs_resid, cmap="YlOrRd",
               s=5, alpha=0.55, linewidths=0, rasterized=True, zorder=2)
    ax.axhline(0, color=PALETTE["diagonal"], lw=1.3, ls="--")
    ax.set_xlabel("Predicted value");  ax.set_ylabel("Residual (pred − true)")
    ax.set_title("Residuals vs. Predicted")
    stat_box(ax, f"R²    {r2:.4f}\nRMSE  {rmse:.4f}", loc="upper right")

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=PALETTE["bg"])
    print(f"[regression] saved → {out_path}")
    if show:
        plt.show()
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--binary",     default=None, help="Path to binary eval CSV.")
    p.add_argument("--regression", default=None, help="Path to regression eval CSV.")
    p.add_argument("--out-binary", default=None)
    p.add_argument("--out-reg",    default=None)
    p.add_argument("--dpi",        type=int, default=200)
    p.add_argument("--no-show",    action="store_true")
    return p.parse_args()

def stem(path):
    return os.path.splitext(os.path.basename(path))[0]

def main():
    args = parse_args()
    show = not args.no_show

    if args.binary:
        df  = pd.read_csv(args.binary)
        out = args.out_binary or f"{stem(args.binary)}_fig.png"
        plot_binary(df, out, dpi=args.dpi, show=show)

        s = stem(args.binary)
        plot_scatter_highlight(df, f"{s}_highlight_pos.png",
                               highlight="pos", dpi=args.dpi, show=show)
        plot_scatter_highlight(df, f"{s}_highlight_neg.png",
                               highlight="neg", dpi=args.dpi, show=show)

    if args.regression:
        df  = pd.read_csv(args.regression)
        out = args.out_reg or f"{stem(args.regression)}_fig.png"
        plot_regression(df, out, dpi=args.dpi, show=show)

    if not args.binary and not args.regression:
        print("Nothing to do. Pass --binary and/or --regression.")

if __name__ == "__main__":
    main()
