"""Turn results/results.csv into report-ready tables (Markdown + LaTeX) and figures (PNG, 300 dpi).

    python scripts/make_tables.py --main-test cifake

Outputs in results/:
    table_main.md / .tex            all metrics, clean test set
    table_generalization.md / .tex  accuracy + AUC on each test set (other generators)
    table_robustness.md / .tex      accuracy under each perturbation
    fig_robustness.png              grouped bars: accuracy per perturbation per method
    fig_generalization.png          grouped bars: AUC per test set per method
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.plotstyle import INK2, METHOD_COLORS, METHOD_LABELS, METHOD_ORDER, setup  # noqa: E402
from src.perturb import DEFAULT_SUITE  # noqa: E402

PERT_LABELS = {"none": "Clean", "jpeg:95": "JPEG 95", "jpeg:75": "JPEG 75", "jpeg:50": "JPEG 50",
               "blur:1": "Blur σ=1", "blur:2": "Blur σ=2", "resize:0.5": "Resize ½", "noise:8": "Noise σ=8"}


def ordered_methods(df):
    present = list(df["method"].unique())
    return [m for m in METHOD_ORDER if m in present] + [m for m in present if m not in METHOD_ORDER]


def save_table(df, path_stem, caption, label, bold_max=True):
    head = "| " + " | ".join([df.index.name or ""] + [str(c) for c in df.columns]) + " |"
    lines = [head, "|" + "---|" * (len(df.columns) + 1)]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)
                                                      for v in row.values]) + " |")
    Path(f"{path_stem}.md").write_text("\n".join(lines) + "\n")
    fmt = df.copy()
    fmt.index.name = None
    for c in fmt.columns:
        if pd.api.types.is_numeric_dtype(fmt[c]):
            col = fmt[c]
            best = col.max()
            fmt[c] = [(f"\\textbf{{{v:.3f}}}" if bold_max and np.isclose(v, best) else f"{v:.3f}")
                      if pd.notna(v) else "--" for v in col]
    tex = fmt.to_latex(escape=False, column_format="l" + "c" * len(fmt.columns))
    tex = ("\\begin{table}[t]\n\\centering\n\\caption{" + caption + "}\n\\label{" + label + "}\n"
           "\\resizebox{\\columnwidth}{!}{%\n" + tex + "}\n\\end{table}\n")
    Path(f"{path_stem}.tex").write_text(tex)


def grouped_bars(ax, pivot, methods, ylabel, ylim=(0, 1.14), label_values=True):
    x = np.arange(len(pivot.index))
    w = 0.8 / len(methods)
    for i, m in enumerate(methods):
        vals = pivot[m].values if m in pivot else np.full(len(x), np.nan)
        bars = ax.bar(x + (i - (len(methods) - 1) / 2) * w, vals, w * 0.92, color=METHOD_COLORS.get(m, "#888"),
                      label=METHOD_LABELS.get(m, m), linewidth=0)
        if label_values:
            for b, v in zip(bars, vals):
                if np.isfinite(v):
                    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom",
                            fontsize=6, color=INK2, rotation=90)
    ax.set_xticks(x)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.legend(ncol=len(methods), loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7,
              handlelength=1.2, columnspacing=1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/results.csv")
    ap.add_argument("--main-test", default="cifake", help="test_set name used for the main + robustness tables")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--exclude", nargs="*", default=["dummy"], help="test sets to leave out of the tables")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt = setup()

    df = pd.read_csv(a.results)
    df = df[~df.test_set.isin(a.exclude)]
    # if an experiment was re-run, keep only the latest row
    df = df.drop_duplicates(["method", "train_set", "test_set", "perturbation"], keep="last")
    methods = ordered_methods(df)
    names = {m: METHOD_LABELS.get(m, m) for m in methods}

    # ---- 1. main table
    main = df[(df.test_set == a.main_test) & (df.perturbation == "none")].set_index("method")
    if len(main):
        cols = ["accuracy", "precision", "recall", "f1", "auc", "real_acc", "fake_acc"]
        t = main.loc[[m for m in methods if m in main.index], cols].rename(index=names)
        t.columns = ["Acc", "Prec", "Recall", "F1", "AUC", "Real acc", "Fake acc"]
        save_table(t, out / "table_main", f"Detection results on the clean {a.main_test} test set.", "tab:main")
        print("\nMAIN\n", t.round(3).to_string())

    # ---- 2. generalisation across test sets
    clean = df[df.perturbation == "none"]
    if clean.test_set.nunique() > 1:
        tests = [a.main_test] + sorted(t for t in clean.test_set.unique() if t != a.main_test)
        acc = clean.pivot(index="test_set", columns="method", values="accuracy").reindex(tests)
        auc = clean.pivot(index="test_set", columns="method", values="auc").reindex(tests)
        g = pd.concat({"Acc": acc[[m for m in methods if m in acc]], "AUC": auc[[m for m in methods if m in auc]]},
                      axis=1)
        g.columns = [f"{names[m]} {k}" for k, m in g.columns]
        save_table(g, out / "table_generalization",
                   "Accuracy and AUC when testing on images from other generators.", "tab:general")
        print("\nGENERALIZATION\n", g.round(3).to_string())
        fig, ax = plt.subplots(figsize=(3.5, 2.6))
        grouped_bars(ax, auc, methods, "ROC-AUC")
        ax.set_xticklabels(tests, rotation=30, ha="right")
        ax.axhline(0.5, color=INK2, lw=0.8, ls="--")
        ax.text(len(tests) - 0.5, 0.51, "chance", fontsize=7, color=INK2, ha="right", va="bottom")
        fig.savefig(out / "fig_generalization.png")
        plt.close(fig)

    # ---- 3. robustness
    rob = df[df.test_set == a.main_test]
    if rob.perturbation.nunique() > 1:
        perts = [p for p in DEFAULT_SUITE if p in set(rob.perturbation)] + \
                sorted(p for p in rob.perturbation.unique() if p not in DEFAULT_SUITE)
        pv = rob.pivot(index="perturbation", columns="method", values="accuracy").reindex(perts)
        r = pv[[m for m in methods if m in pv]].rename(index=PERT_LABELS, columns=names)
        save_table(r, out / "table_robustness", f"Accuracy on {a.main_test} under common image degradations.",
                   "tab:robust")
        print("\nROBUSTNESS\n", r.round(3).to_string())
        fig, ax = plt.subplots(figsize=(7.0, 2.6))  # full IEEE text width (figure*)
        grouped_bars(ax, pv, methods, "Accuracy")
        ax.set_xticklabels([PERT_LABELS.get(p, p) for p in perts])
        ax.axhline(0.5, color=INK2, lw=0.8, ls="--")
        fig.savefig(out / "fig_robustness.png")
        plt.close(fig)

    print(f"\nwrote tables and figures to {out}/")


if __name__ == "__main__":
    main()
