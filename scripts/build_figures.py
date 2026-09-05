#!/usr/bin/env python3
"""Build the eight paper figures from stored machine-readable results."""

from __future__ import annotations

import ast
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "figures"
DERIVED = OUT / "values"

CORE = RESULTS / "seven_dataset_core.csv"
GROUPS = RESULTS / "independent_group_responses.csv"
LINEAGE = RESULTS / "evidence_class_summary.csv"
UNIQUE_SCALER = RESULTS / "unique_acquisition_scaler.csv"
PATHWAY = RESULTS / "training_validation_decomposition.csv"
TYPE_STRESS = RESULTS / "type_multiplicity_stress.csv"
TEMPORAL = RESULTS / "temporal_probabilities.csv.gz"
MATERIALIZATION = RESULTS / "materialization_benchmark.csv"
MATERIALIZATION_FOLDS = RESULTS / "materialization_fold_responses.csv"
IOU_COUNTEREXAMPLE = RESULTS / "iou_matching_counterexample.csv"

ORDER = [
    "hapt",
    "mhealth",
    "synthetic",
    "daphnet",
    "harth",
    "har70",
    "ppg_dalia",
]
DISPLAY = {
    "hapt": "HAPT",
    "mhealth": "MHEALTH",
    "synthetic": "Synthetic",
    "daphnet": "Daphnet",
    "harth": "HARTH",
    "har70": "HAR70+",
    "ppg_dalia": "PPG-DaLiA",
}

BLUE = "#2F6FA3"
ORANGE = "#D97826"
TEAL = "#168A86"
PURPLE = "#7651A8"
DARK = "#33383F"
MID = "#6B7280"
LIGHT = "#D5D9DE"
PALE_BLUE = "#E8F0F8"
PALE_ORANGE = "#FAEEE4"
PALE_TEAL = "#E6F3F1"
PALE_RED = "#F9E9E9"

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.0,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.7,
        "savefig.transparent": False,
    }
)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.04,
        1.04,
        label,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=10,
        va="bottom",
        ha="left",
    )


def clean_axis(ax: plt.Axes, grid: str | None = "x") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis=grid, color="#E7E9EC", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, filename: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUT / filename, bbox_inches="tight", facecolor="white",
        metadata={"Creator": "Matplotlib", "Author": None,
                  "CreationDate": None, "ModDate": None},
    )
    plt.close(fig)



def architecture_display(value: str) -> str:
    return "BiGRU" if value.lower() == "bigru" else value.upper()


def rounded(value: float) -> str:
    return f"{value:+.3f}".replace("+0.000", "0.000").replace("-0.000", "0.000")


def validate_inputs() -> None:
    required = [
        CORE,
        GROUPS,
        LINEAGE,
        UNIQUE_SCALER,
        PATHWAY,
        TYPE_STRESS,
        TEMPORAL,
        MATERIALIZATION,
        MATERIALIZATION_FOLDS,
        IOU_COUNTEREXAMPLE,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("\n".join(missing))

    core = pd.read_csv(CORE).set_index("dataset").loc[ORDER]
    expected_core = {
        "hapt": -0.094,
        "mhealth": -0.112,
        "synthetic": -0.020,
        "daphnet": -0.027,
        "harth": -0.103,
        "har70": 0.004,
        "ppg_dalia": -0.228,
    }
    observed = core["typed_signed_response"].round(3).to_dict()
    if observed != expected_core:
        raise AssertionError(f"Core anchors disagree: {observed}")

    unique = pd.read_csv(UNIQUE_SCALER).set_index("dataset")
    expected_unique = {
        "hapt": -0.129,
        "mhealth": -0.123,
        "synthetic": -0.039,
        "ppg_dalia": -0.229,
    }
    observed_unique = unique["signed_response"].round(3).to_dict()
    if observed_unique != expected_unique:
        raise AssertionError(f"Scaler anchors disagree: {observed_unique}")

    pathway = pd.read_csv(PATHWAY).set_index(["dataset", "condition"])
    expected_probability = {
        ("hapt", "train_only"): 0.551,
        ("hapt", "validation_only"): 0.641,
        ("hapt", "both"): 0.737,
        ("mhealth", "train_only"): 0.812,
        ("mhealth", "validation_only"): 0.698,
        ("mhealth", "both"): 0.807,
        ("ppg_dalia", "train_only"): 0.887,
        ("ppg_dalia", "validation_only"): 0.879,
        ("ppg_dalia", "both"): 0.766,
    }
    observed_probability = {
        key: round(float(pathway.loc[key, "maximum_probability_response"]), 3)
        for key in expected_probability
    }
    if observed_probability != expected_probability:
        raise AssertionError(
            f"Pathway probability anchors disagree: {observed_probability}"
        )

    operational = pd.read_csv(MATERIALIZATION)
    ordinary = operational[
        operational["role"].eq("ordinary")
        & operational["transformation"].ne("clean")
    ]
    if len(ordinary) != 18:
        raise AssertionError(f"Expected 18 ordinary conditions, found {len(ordinary)}")
    f1_changed = ordinary["segment_f1_response_mean"].round(3).ne(0).sum()
    probability_changed = ordinary["maximum_probability_response"].round(3).ne(0).sum()
    both_null = (
        ordinary["segment_f1_response_mean"].round(3).eq(0)
        & ordinary["maximum_probability_response"].round(3).eq(0)
    ).sum()
    if (f1_changed, probability_changed, both_null) != (12, 13, 5):
        raise AssertionError(
            "Operational counts disagree at three-decimal reporting precision: "
            f"{f1_changed}, {probability_changed}, {both_null}"
        )
    reorder = ordinary[
        ordinary["transformation"].eq("deterministic_row_reorder")
    ]
    if not (
        reorder["segment_f1_response_mean"].round(3).eq(0).all()
        and reorder["maximum_probability_response"].round(3).eq(0).all()
    ):
        raise AssertionError("Row-reordering control is not null at reported precision")
    quotient = operational[
        operational["role"].eq("quotient")
        & operational["transformation"].ne("clean")
    ]
    if not (
        quotient["segment_f1_response_mean"].eq(0).all()
        and quotient["maximum_probability_response"].eq(0).all()
    ):
        raise AssertionError("A quotient operational response is nonzero")
    if round(float(ordinary["maximum_probability_response"].max()), 3) != 0.587:
        raise AssertionError("Maximum operational probability response disagrees")


def draw_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str = "white",
    edgecolor: str = LIGHT,
    fontsize: float = 7.0,
    weight: str = "normal",
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=0.8,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
    )


def compiler_overview() -> None:
    """Illustrate the same five evidence units in two row materializations."""
    fig = plt.figure(figsize=(7.2, 3.95))
    a = fig.add_axes([0.014, 0.643, 0.972, 0.340])
    b = fig.add_axes([0.014, 0.028, 0.472, 0.575])
    c = fig.add_axes([0.514, 0.028, 0.472, 0.575])
    for ax, letter, heading in (
        (a, "A", "Same evidence, different row materializations"),
        (b, "B", "Row multiplicity changes\npreprocessing and risk weights"),
        (c, "C", "Compilation gives the same\nevidence units"),
    ):
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")
        ax.add_patch(FancyBboxPatch(
            (0.002, 0.002), 0.996, 0.996,
            boxstyle="round,pad=0.009,rounding_size=0.015",
            linewidth=0.9, edgecolor="#C9D2DE", facecolor="white",
            clip_on=False, zorder=-10,
        ))
        ax.text(0.018, 0.963, letter, weight="bold", fontsize=10, va="top")
        ax.text(0.06 if ax is a else 0.078, 0.94, heading,
                weight="bold", fontsize=8.45, va="top")
    columns = ["acq.", "src.", "emit.", "kind", "support", "n"]
    rows = [
        ["a", "S1", "e1", "range", "[20, 40]", "1"],
        ["a", "S1", "e2", "timestamp", "31", "1"],
        ["a", "S2", "e3", "range", "[18, 43]", "1"],
        ["b", "S1", "e4", "negative", "all", "1"],
        ["b", "S2", "e5", "bag", "[0, 60]", "1"],
    ]
    for left, title, copies in (
        (0.03, "One-copy materialization  D¹", False),
        (0.515, "Five-copy materialization  D⁵", True),
    ):
        a.text(left, 0.755, title, fontsize=7.35, weight="bold")
        materialized = [row[:] for row in rows]
        if copies:
            materialized[0][-1] = "×5"
        table = a.table(
            cellText=materialized, colLabels=columns, cellLoc="center",
            colLoc="center", bbox=[left, 0.21, 0.455, 0.50],
            colWidths=[0.07, 0.07, 0.10, 0.16, 0.14, 0.065],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(6.05)
        for (ri, ci), cell in table.get_celld().items():
            cell.set_edgecolor("#CBD4DF")
            cell.set_linewidth(0.55)
            if ri == 0:
                cell.set_facecolor("#F0F3F7")
                cell.get_text().set_fontweight("bold")
            elif ri == 1:
                cell.set_facecolor(PALE_ORANGE)
                if ci == 5 and copies:
                    cell.get_text().set_color(ORANGE)
                    cell.get_text().set_fontweight("bold")
    a.text(0.5, 0.075,
           "The same five evidence units are stored as five versus nine rows.",
           ha="center", va="center", fontsize=7.1, weight="bold", color=DARK)

    # Both comparisons use the same vertical geometry and unchanged fractions.
    height = 0.112
    for center, title_y, title, font in (
        (0.60, 0.701, "Range risk mass within acquisition a", 7.7),
        (0.227, 0.328, "Acquisition scaler mass across a and b", 7.3),
    ):
        b.text(0.08, title_y, title, weight="bold", fontsize=font)
        b.annotate("", (0.6855, center), (0.5645, center),
                   arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.65,
                                   mutation_scale=11.2))
        b.text(0.325, center - 0.122, "D¹", ha="center", fontsize=8.48)
        b.text(0.85, center - 0.122, "D⁵", ha="center", fontsize=8.48)
    bars = [
        (0.12, 0.60, 0.18, ORANGE, "1/3", 8.48, True),
        (0.30, 0.60, 0.23, "#AEB8C5", "other 2/3", 8.48, False),
        (0.72, 0.60, 0.20, ORANGE, "5/7", 8.48, True),
        (0.92, 0.60, 0.06, "#AEB8C5", "2/7", 6.36, False),
        (0.12, 0.227, 0.245, BLUE, "a: 3/5", 8.48, True),
        (0.365, 0.227, 0.165, TEAL, "b: 2/5", 8.48, False),
        (0.72, 0.227, 0.202, BLUE, "a: 7/9", 8.48, True),
        (0.922, 0.227, 0.058, TEAL, "", 6.254, False),
    ]
    for x, y, width, color, label, size, bold in bars:
        b.add_patch(Rectangle((x, y - height / 2), width, height, color=color))
        if label:
            b.text(x + width / 2, y, label, color="white", ha="center",
                   va="center", fontsize=size, weight="bold" if bold else "normal")
    b.annotate("b: 2/9", xy=(0.951, 0.276), xytext=(0.951, 0.313),
               ha="center", va="bottom", fontsize=6.254, color=DARK,
               arrowprops=dict(arrowstyle="-", color=TEAL, lw=0.7))

    draw_box(c, (0.06, 0.67), 0.18, 0.13, "D¹\n5 rows",
             facecolor=PALE_BLUE, edgecolor=BLUE, fontsize=7.42, weight="bold")
    draw_box(c, (0.06, 0.45), 0.18, 0.13, "D⁵\n9 rows",
             facecolor=PALE_ORANGE, edgecolor=ORANGE, fontsize=7.42, weight="bold")
    draw_box(c, (0.35, 0.54), 0.29, 0.20, "Compiler\nQ(·)",
             facecolor="#EDF6EF", edgecolor="#397A51", fontsize=8.0, weight="bold")
    draw_box(c, (0.73, 0.55), 0.22, 0.18, "5 evidence units\nacq. a: 3\nacq. b: 2",
             facecolor=PALE_TEAL, edgecolor=TEAL, fontsize=6.4, weight="bold")
    for start in ((0.24, 0.735), (0.24, 0.515)):
        c.annotate("", (0.35, 0.64), start,
                   arrowprops=dict(arrowstyle="->", color=MID, lw=1.1))
    c.annotate("", (0.73, 0.64), (0.64, 0.64),
               arrowprops=dict(arrowstyle="->", color="#397A51", lw=1.5))
    c.text(0.5, 0.35, "Q(D¹) = Q(D⁵)", ha="center", va="center",
           fontsize=11.13, weight="bold", color=DARK)
    c.text(0.5, 0.25, "Same scaler input · same typed risk", ha="center", fontsize=7.35)
    for y, label in ((0.145, "construct key → group rows → canonical records"),
                     (0.045, "fit scaler → typed risk → select checkpoint → decode")):
        c.text(0.5, y, label, ha="center", va="center", fontsize=6.6, color=DARK,
               bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor=LIGHT))
    save_figure(fig, "semantic_contract.pdf")

def evidence_class_composition() -> None:
    data = pd.read_csv(LINEAGE)
    stress = (
        data[data["condition"].eq("five_copy_range_materialization")]
        .set_index("dataset")
        .loc[ORDER]
    )
    retained = 100 * stress["evidence_classes"] / stress["serialized_rows"]
    redundant = 100 - retained
    y = np.arange(len(ORDER))
    fig, ax = plt.subplots(figsize=(7.2, 3.25))
    ax.barh(y, retained, color=BLUE, height=0.62, label="Evidence units")
    ax.barh(y, redundant, left=retained, color=ORANGE, height=0.62, label="Additional materializations")
    for pos, dataset in enumerate(ORDER):
        row = stress.loc[dataset]
        ratio = row["serialized_rows"] / row["evidence_classes"]
        ax.text(
            min(2.0, retained.iloc[pos] * 0.03),
            pos,
            f"{int(row['serialized_rows'])} stored → {int(row['evidence_classes'])} evidence units ({ratio:.2f}×)",
            va="center",
            ha="left",
            color="white",
            fontsize=7.0,
            fontweight="bold",
        )
    ax.axhline(5.5, color="#B9BEC6", linewidth=0.8)
    ax.set_yticks(y, [DISPLAY[item] for item in ORDER])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Composition of five-copy table (%)")
    ax.legend(
        frameon=False,
        ncol=2,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        borderaxespad=0,
        columnspacing=1.2,
    )
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#E7E9EC", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.14, right=0.77, top=0.84, bottom=0.18)
    save_figure(fig, "lineage_audit.pdf")

def draw_reference_lines(ax: plt.Axes) -> None:
    ax.axvline(0, color=DARK, linewidth=0.9, zorder=1)

def plot_group_responses(ax: plt.Axes, *, show_panel_label: bool) -> None:
    groups = pd.read_csv(GROUPS)
    groups = groups[groups["method"].eq("ordinary_learning")]
    rng = np.random.default_rng(27)
    y = np.arange(len(ORDER))
    for pos, dataset in enumerate(ORDER):
        values = (
            groups.loc[groups["dataset"].eq(dataset), "signed_response"]
            .astype(float)
            .to_numpy()
        )
        jitter = rng.uniform(-0.16, 0.16, len(values))
        color = PURPLE if dataset == "ppg_dalia" else BLUE
        ax.scatter(values, pos + jitter, s=12, alpha=0.72, color=color, edgecolors="none", zorder=2)
        mean = float(np.mean(values))
        median = float(np.median(values))
        ax.scatter(mean, pos, marker="D", s=32, color=ORANGE, edgecolor="white", linewidth=0.5, zorder=4)
        ax.plot([median, median], [pos - 0.22, pos + 0.22], color="#171A1F", linewidth=2.2, solid_capstyle="butt", zorder=3)
        ax.text(
            1.015,
            pos,
            f"n={len(values)}  mean {mean:+.3f}  median {median:+.3f}",
            transform=ax.get_yaxis_transform(),
            va="center",
            ha="left",
            fontsize=6.7,
            clip_on=False,
        )
    draw_reference_lines(ax)
    ax.set_yticks(y, [DISPLAY[item] for item in ORDER])
    ax.invert_yaxis()
    ax.set_xlabel("Ordinary segment-F1 response (five-copy − clean)")
    ax.set_xlim(-0.73, 0.44)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#E7E9EC", linewidth=0.6)
    if show_panel_label:
        panel_label(ax, "C")


def core_response() -> None:
    core = pd.read_csv(CORE).set_index("dataset").loc[ORDER]
    fig = plt.figure(figsize=(7.2, 3.35))
    a = fig.add_axes([0.12, 0.205, 0.22, 0.64])
    b = fig.add_axes([0.47, 0.205, 0.36, 0.64])
    y = np.arange(len(ORDER))
    a.scatter(core["typed_signed_response"], y, s=36, color=BLUE, label="Ordinary learning")
    a.scatter(core["quotient_signed_response"], y, s=40, facecolors="none",
              edgecolors=TEAL, linewidths=1.2, label="Quotient learning", zorder=3)
    draw_reference_lines(a)
    a.set_yticks(y, [DISPLAY[ds] for ds in ORDER])
    a.set_ylim(len(ORDER) - 0.5, -0.5)
    a.set_xlim(-0.255, 0.055)
    a.set_xticks([-0.2, -0.1, 0])
    a.set_xlabel("Segment-F1 response\n(five-copy − clean)")
    a.set_title("A  Dataset-level response", loc="left", weight="bold", fontsize=8.6)
    clean_axis(a)
    plot_group_responses(b, show_panel_label=False)
    b.set_title("B  Held-group ordinary responses", loc="left", weight="bold", fontsize=8.6)
    # Only the zero reference is part of the submitted figure.
    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="upper left",
               bbox_to_anchor=(0.11, 1.03), columnspacing=1.0)
    group_handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=5.5, color=BLUE, label="Held group"),
        Line2D([], [], marker="o", linestyle="none", markersize=5.5, color=PURPLE, label="PPG-DaLiA held group"),
        Line2D([], [], marker="D", linestyle="none", markersize=5.5, color=ORANGE, label="Mean"),
        Line2D([], [], marker="|", linestyle="none", markersize=10, markeredgewidth=2.2, color="#171A1F", label="Median"),
    ]
    fig.legend(handles=group_handles, frameon=False, ncol=4, loc="lower center",
               bbox_to_anchor=(0.65, -0.015), columnspacing=1.0, handletextpad=0.4)
    save_figure(fig, "core_response.pdf")



def transmission_pathways() -> None:
    core = pd.read_csv(CORE).set_index("dataset")
    unique = pd.read_csv(UNIQUE_SCALER).set_index("dataset")
    roles_data = pd.read_csv(PATHWAY).set_index(["dataset","condition"])
    fig = plt.figure(figsize=(7.2,4.00))
    a = fig.add_axes([0.092,0.145,0.228,0.605])
    starts = [0.402,0.595,0.788]
    facets = [fig.add_axes([x,0.145,0.186,0.605]) for x in starts]
    data_a = ["hapt","mhealth","synthetic","ppg_dalia"]
    for y,ds in enumerate(data_a):
        row,uq = float(core.loc[ds,"typed_signed_response"]),float(unique.loc[ds,"signed_response"])
        offset = 0.055 if abs(row-uq)<0.003 else 0.0
        a.plot([row,uq],[y-offset,y+offset],color="#AAB3BE",lw=0.85,zorder=1)
        a.scatter(row,y-offset,s=56,color=BLUE,edgecolors="white",linewidths=0.5,zorder=4)
        a.scatter(uq,y+offset,s=70,facecolors="none",edgecolors=TEAL,linewidths=1.45,zorder=5)
    a.axvspan(-0.02,0.02,color="#B8C1CB",alpha=0.045,lw=0,zorder=0)
    a.axvline(0,color=DARK,lw=1.15,zorder=1)
    a.set_xlim(-0.255,0.035)
    a.set_ylim(3.5,-0.5)
    a.set_yticks(range(4),[DISPLAY[x] for x in data_a])
    a.set_xticks([-0.2,-0.1,0])
    a.set_xlabel("Signed segment-F1 response",fontsize=9.4,labelpad=6)
    clean_axis(a)
    for ax in [a,*facets]:
        ax.tick_params(labelsize=8.5)
    roles = [("train_only","Training only","o",BLUE),
             ("validation_only","Validation only","D",PURPLE),
             ("both","Both roles","s",DARK)]
    for j,(ax,ds) in enumerate(zip(facets,["hapt","mhealth","ppg_dalia"])):
        ax.axvspan(-0.02,0.02,color="#B8C1CB",alpha=0.045,lw=0,zorder=0)
        ax.axvline(0,color=DARK,lw=1.15,zorder=1)
        for condition,label,marker,color in roles:
            r=roles_data.loc[(ds,condition)]
            ax.scatter(r["segment_f1_response_mean"],r["maximum_probability_response"],
                       marker=marker,color=color,s=88,edgecolors="white",linewidths=0.55,zorder=4)
        # Keep the complete marker at the stored PPG both-role x=-0.218639...
        # visible. All three facets retain identical limits and tick values.
        ax.set_xlim(-0.24,0.14)
        ax.set_ylim(0,0.95)
        ax.set_xticks([-0.2,0,0.1])
        ax.set_yticks([0,0.2,0.4,0.6,0.8])
        ax.set_title(DISPLAY[ds],fontsize=10.3,fontweight="bold",pad=8)
        clean_axis(ax,grid="y")
        if j:
            ax.tick_params(axis="y",left=False,labelleft=False)
            ax.spines["left"].set_visible(False)
        else:
            ax.set_ylabel(r"Maximum held-out $|\Delta p|$",fontsize=9.4,labelpad=6)
    fig.text(0.092,0.966,"A  Sensitivity with\n    acquisition-level scaling",ha="left",va="top",fontsize=10.3,fontweight="bold",linespacing=1.04)
    fig.text(0.402,0.966,"B  Training and validation materialization",ha="left",va="top",fontsize=10.3,fontweight="bold")
    fig.legend(handles=[Line2D([],[],marker="o",linestyle="none",color=BLUE,markersize=6.1,label="Row-fitted scaler"),
                        Line2D([],[],marker="o",linestyle="none",markerfacecolor="none",markeredgecolor=TEAL,markeredgewidth=1.3,markersize=6.8,label="Unique-acquisition scaler")],
               loc="upper left",bbox_to_anchor=(0.084,0.865),fontsize=8.2,frameon=False,handletextpad=0.4,borderaxespad=0,labelspacing=0.4)
    fig.legend(handles=[Line2D([],[],marker=m,linestyle="none",color=c,markersize=6.4,label=l) for _,l,m,c in roles],
               loc="upper left",bbox_to_anchor=(0.394,0.895),ncol=3,fontsize=8.2,frameon=False,columnspacing=0.7,handletextpad=0.3,borderaxespad=0)
    fig.text(0.688,0.035,"Signed segment-F1 response",ha="center",va="bottom",fontsize=9.4)
    for j,ds in enumerate(data_a):
        assert float(a.collections[2*j].get_offsets()[0,0]) == float(core.loc[ds,"typed_signed_response"])
        assert float(a.collections[2*j+1].get_offsets()[0,0]) == float(unique.loc[ds,"signed_response"])
    for ax,ds in zip(facets,["hapt","mhealth","ppg_dalia"]):
        for artist,(condition,_,_,_) in zip(ax.collections,roles):
            expected = roles_data.loc[(ds,condition),["segment_f1_response_mean","maximum_probability_response"]].to_numpy(float)
            np.testing.assert_array_equal(artist.get_offsets()[0],expected)
    save_figure(fig, "transmission_pathways.pdf")

TRANSFORMATION_LABELS = {
    "one_to_many_metadata_join": "One-to-many metadata join",
    "repeated_export_concatenation": "Repeated export",
    "exploded_list_metadata": "Exploded list metadata",
    "heterogeneous_metadata_fanout": "Heterogeneous metadata fan-out",
    "deterministic_row_reorder": "Row reorder",
    "mixed_materialization_sequence": "Mixed sequence",
}


def materialization_benchmark() -> None:
    publication = pd.read_csv(MATERIALIZATION)
    folds = pd.read_csv(MATERIALIZATION_FOLDS)
    ordinary = publication[publication.role.eq("ordinary") & publication.transformation.ne("clean")]
    fig,axes = plt.subplots(2,3,figsize=(7.2,5.20),sharey=True)
    operations=list(TRANSFORMATION_LABELS)
    y=np.arange(6)
    counts=[]
    for col,(ds,arch) in enumerate([("hapt","tcn"),("mhealth","tcn"),("hapt","bigru")]):
        pub=ordinary[ordinary.dataset.eq(ds)&ordinary.architecture.eq(arch)].set_index("transformation").loc[operations]
        method="ordinary_learning_bigru" if arch=="bigru" else "ordinary_learning"
        ff=folds[folds.dataset.eq(ds)&folds.architecture.eq(arch)&folds.method.eq(method)]
        for row,(fold_field,pub_field) in enumerate([("segment_f1_response","segment_f1_response_mean"),("maximum_absolute_probability_response","maximum_probability_response")]):
            ax=axes[row,col]
            for pos,op in enumerate(operations):
                values=ff.loc[ff.transformation.eq(op),fold_field].to_numpy(float)
                assert len(values)==3
                ax.plot([values.min(),values.max()],[pos,pos],color=BLUE,alpha=0.53,lw=0.85,zorder=2)
                ax.scatter(values,np.full(3,pos),s=11,color=BLUE,alpha=0.20,edgecolors="none",zorder=3,clip_on=False)
                counts.append(len(values))
            ax.scatter(pub[pub_field],y,s=27,color=BLUE,edgecolor="white",linewidth=0.55,zorder=5,clip_on=False)
            for pos,op in enumerate(operations):
                expected = ff.loc[ff.transformation.eq(op),fold_field].to_numpy(float)
                np.testing.assert_array_equal(ax.collections[pos].get_offsets()[:,0],expected)
            np.testing.assert_array_equal(ax.collections[-1].get_offsets()[:,0],pub[pub_field].to_numpy(float))
            ax.axvline(0,color=DARK,lw=1.1,zorder=1)
            ax.set_xlim((-0.22,0.20) if row==0 else (0,0.62))
            ax.set_xticks([-0.2,-0.1,0,0.1,0.2] if row==0 else [0,0.2,0.4,0.6])
            ax.set_ylim(5.5,-0.5)
            ax.set_yticks(y,list(TRANSFORMATION_LABELS.values()))
            ax.set_title(f"{DISPLAY[ds]}–{architecture_display(arch)}",fontsize=8.3,fontweight="bold",pad=6)
            clean_axis(ax,grid="y")
            ax.grid(axis="y",color="#E7E9EC",linewidth=0.5)
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y",left=col==0,labelleft=col==0,labelsize=8.1)
            ax.tick_params(axis="x",labelsize=7.2)
    fig.subplots_adjust(left=0.37,right=0.99,bottom=0.12,top=0.895,wspace=0.16,hspace=0.69)
    fig.text(0.016,0.985,"A  Signed segment-F1 response",fontsize=9.4,fontweight="bold",va="top")
    fig.text(0.016,0.500,"B  Maximum absolute held-out probability response",fontsize=9.4,fontweight="bold",va="top")
    fig.text(0.68,0.553,"Signed segment-F1 response",fontsize=8.5,ha="center",va="top")
    fig.text(0.68,0.036,r"Maximum held-out $|\Delta p|$",fontsize=8.5,ha="center",va="bottom")
    save_figure(fig, "materialization_benchmark.pdf")

def binary_intervals(values: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(values, dtype=bool)
    padded = np.pad(values.astype(np.int8), (1, 1))
    starts = np.flatnonzero(np.diff(padded) == 1)
    stops = np.flatnonzero(np.diff(padded) == -1)
    return list(zip(starts.tolist(), stops.tolist(), strict=True))


def interval_iou(left: tuple[int, int], right: tuple[int, int]) -> float:
    intersection = max(0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return intersection / union if union else 0.0


def maximum_cardinality(
    truth: list[tuple[int, int]],
    prediction: list[tuple[int, int]],
    threshold: float = 0.5,
) -> int:
    adjacency = [
        [j for j, predicted in enumerate(prediction) if interval_iou(reference, predicted) >= threshold]
        for reference in truth
    ]
    matched_prediction: dict[int, int] = {}

    def augment(reference_index: int, seen: set[int]) -> bool:
        for prediction_index in adjacency[reference_index]:
            if prediction_index in seen:
                continue
            seen.add(prediction_index)
            prior = matched_prediction.get(prediction_index)
            if prior is None or augment(prior, seen):
                matched_prediction[prediction_index] = reference_index
                return True
        return False

    return sum(augment(index, set()) for index in range(len(truth)))


def sequence_f1(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth_intervals = binary_intervals(truth)
    prediction_intervals = binary_intervals(prediction)
    matches = maximum_cardinality(truth_intervals, prediction_intervals)
    precision = matches / len(prediction_intervals) if prediction_intervals else 0.0
    recall = (
        matches / len(truth_intervals)
        if truth_intervals
        else (1.0 if not prediction_intervals else 0.0)
    )
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def derive_temporal_examples() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(TEMPORAL)
    frame = frame[frame["dataset"].eq("hapt")].copy()
    clean_name = "clean"
    stress_name = "five_copy_range_materialization"
    typed = "ordinary_learning"
    quotient = "quotient_learning"
    rows: list[dict[str, object]] = []
    for sequence_id, local in frame.groupby("sequence_id", sort=True):
        pivot = local.pivot_table(
            index="bin",
            columns=["method", "condition"],
            values="probability",
            aggfunc="first",
        )
        required = [
            (typed, clean_name),
            (typed, stress_name),
            (quotient, clean_name),
            (quotient, stress_name),
        ]
        if any(column not in pivot.columns for column in required):
            continue
        truth = (
            local[
                local["method"].eq(typed)
                & local["condition"].eq(clean_name)
            ]
            .sort_values("bin")
            .drop_duplicates("bin")["truth"]
            .to_numpy(dtype=int)
        )
        clean_probability = pivot[(typed, clean_name)].to_numpy()
        stress_probability = pivot[(typed, stress_name)].to_numpy()
        clean_prediction = clean_probability >= 0.5
        stress_prediction = stress_probability >= 0.5
        clean_f1 = sequence_f1(truth, clean_prediction)
        stress_f1 = sequence_f1(truth, stress_prediction)
        rows.append(
            {
                "sequence_id": sequence_id,
                "group": str(local["group"].iloc[0]),
                "fold": int(local["fold"].iloc[0]),
                "max_abs_probability_response": float(np.max(np.abs(stress_probability - clean_probability))),
                "acquisition_f1_clean": clean_f1,
                "acquisition_f1_five_copy": stress_f1,
                "acquisition_f1_response": stress_f1 - clean_f1,
                "changed_decoded_bins": int(np.sum(clean_prediction != stress_prediction)),
                "clean_predicted_segments": len(binary_intervals(clean_prediction)),
                "five_copy_predicted_segments": len(binary_intervals(stress_prediction)),
            }
        )
    sequence_metrics = pd.DataFrame(rows)
    if sequence_metrics.empty:
        raise AssertionError("No complete HAPT temporal sequences found")

    used: set[str] = set()
    candidates: list[tuple[str, pd.Series]] = []
    zero_f1 = sequence_metrics[np.isclose(sequence_metrics["acquisition_f1_response"], 0.0)].sort_values(
        ["max_abs_probability_response", "sequence_id"],
        ascending=[False, True],
    )
    candidates.append(("large probability change; ΔF1 = 0", zero_f1.iloc[0]))
    used.add(str(zero_f1.iloc[0]["sequence_id"]))

    changed = sequence_metrics[
        sequence_metrics["changed_decoded_bins"].gt(0)
        & sequence_metrics["acquisition_f1_response"].abs().gt(1e-12)
        & sequence_metrics["sequence_id"].astype(str).map(lambda value: value not in used)
    ].copy()
    changed["little_f1_key"] = changed["acquisition_f1_response"].abs()
    changed = changed.sort_values(
        ["little_f1_key", "changed_decoded_bins", "max_abs_probability_response", "sequence_id"],
        ascending=[True, False, False, True],
    )
    candidates.append(("decoded bins changed; small ΔF1", changed.iloc[0]))
    used.add(str(changed.iloc[0]["sequence_id"]))

    large_f1 = sequence_metrics[
        sequence_metrics["sequence_id"].astype(str).map(lambda value: value not in used)
    ].copy()
    large_f1["abs_f1_response"] = large_f1["acquisition_f1_response"].abs()
    large_f1 = large_f1.sort_values(
        ["abs_f1_response", "max_abs_probability_response", "sequence_id"],
        ascending=[False, False, True],
    )
    candidates.append(("large acquisition-level |ΔF1|", large_f1.iloc[0]))

    selected = pd.DataFrame(
        [
            {**series.to_dict(), "selection_role": role}
            for role, series in candidates
        ]
    ).drop(columns=["little_f1_key", "abs_f1_response"], errors="ignore")
    DERIVED.mkdir(parents=True, exist_ok=True)
    sequence_metrics.drop(columns=["little_f1_key", "abs_f1_response"], errors="ignore").to_csv(
        DERIVED / "temporal_sequence_metrics.csv", index=False
    )
    selected.to_csv(DERIVED / "temporal_example_selection.csv", index=False)
    return frame, selected


def draw_lane(ax: plt.Axes, intervals: list[tuple[int, int]], color: str, label: str, n_bins: int) -> None:
    ax.set_xlim(0, n_bins)
    ax.set_ylim(0, 1)
    for start, stop in intervals:
        ax.add_patch(Rectangle((start, 0.18), stop - start, 0.64, color=color, linewidth=0))
    ax.text(-0.012, 0.5, label, transform=ax.transAxes, ha="right", va="center", fontsize=6.1)
    ax.axis("off")


def temporal_examples() -> None:
    frame, selected = derive_temporal_examples()
    clean_name = "clean"
    stress_name = "five_copy_range_materialization"
    typed = "ordinary_learning"
    quotient = "quotient_learning"
    fig = plt.figure(figsize=(7.2, 7.35))
    outer = fig.add_gridspec(3, 1, hspace=0.37)
    styles = [
        ((typed, clean_name), BLUE, "-", "Ordinary clean", 1.3),
        ((typed, stress_name), ORANGE, "-", "Ordinary five-copy", 1.3),
        ((quotient, clean_name), TEAL, "--", "Quotient learning: clean = five-copy", 1.3),
    ]
    for panel_index, row in selected.reset_index(drop=True).iterrows():
        inner = outer[panel_index].subgridspec(5, 1, height_ratios=[6.0, 0.52, 0.52, 0.52, 0.52], hspace=0.05)
        ax = fig.add_subplot(inner[0])
        local = frame[frame["sequence_id"].eq(row["sequence_id"])].copy()
        base = (
            local[
                local["method"].eq(typed)
                & local["condition"].eq(clean_name)
            ]
            .sort_values("bin")
            .drop_duplicates("bin")
        )
        bins = base["bin"].to_numpy(dtype=int)
        times = base["time_seconds"].to_numpy(dtype=float)
        truth = base["truth"].to_numpy(dtype=int)
        ax.fill_between(times, 0, truth, step="mid", color="#E4E6E9", alpha=0.9, label="Reference")
        probabilities: dict[tuple[str, str], np.ndarray] = {}
        for key, color, linestyle, label, width in styles:
            method, condition = key
            curve = (
                local[
                    local["method"].eq(method)
                    & local["condition"].eq(condition)
                ]
                .sort_values("bin")
                .drop_duplicates("bin")
            )
            values = curve["probability"].to_numpy(dtype=float)
            probabilities[key] = values
            ax.plot(times, values, color=color, linestyle=linestyle, linewidth=width, label=label)
        ax.axhline(0.5, color=MID, linestyle=":", linewidth=0.8)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(times.min(), times.max())
        ax.set_ylabel(r"$p(y_t=1)$")
        ax.set_title(
            f"{chr(65 + panel_index)}  {row['selection_role']}",
            loc="left",
            fontweight="bold",
            fontsize=8.6,
        )
        annotation = (
            f"max |Δp| {row['max_abs_probability_response']:.3f}   "
            f"ΔF1 {row['acquisition_f1_response']:+.3f}   "
            f"changed bins {int(row['changed_decoded_bins'])}   "
            f"segments {int(row['clean_predicted_segments'])}→{int(row['five_copy_predicted_segments'])}"
        )
        ax.text(0.995, 0.965, annotation, transform=ax.transAxes, ha="right", va="top", fontsize=6.4, bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
        clean_axis(ax, grid="y")
        lane_specs = [
            (truth.astype(bool), "#858B93", "Reference"),
            (probabilities[(typed, clean_name)] >= 0.5, BLUE, "Ordinary clean"),
            (probabilities[(typed, stress_name)] >= 0.5, ORANGE, "Ordinary five-copy"),
            (probabilities[(quotient, clean_name)] >= 0.5, TEAL, "Quotient"),
        ]
        for lane_index, (binary, color, label) in enumerate(lane_specs, start=1):
            lane = fig.add_subplot(inner[lane_index])
            draw_lane(lane, binary_intervals(binary), color, label, len(binary))
        if panel_index == 0:
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.56, 1.005), columnspacing=0.8)
    fig.supxlabel("Time (s)", x=0.57, y=0.012, fontsize=8.0)
    fig.subplots_adjust(left=0.16, right=0.985, top=0.93, bottom=0.07)
    save_figure(fig, "temporal_examples.pdf")


def type_multiplicity_heatmap() -> None:
    data = pd.read_csv(TYPE_STRESS)
    ordinary = data[data["method"].eq("ordinary_learning")].copy()
    dataset_order = ["hapt", "mhealth", "synthetic"]
    type_order = ["range", "timestamp", "negative"]
    observed_types = list(dict.fromkeys(ordinary["annotation_type"].astype(str)))
    for annotation_type in observed_types:
        if annotation_type not in type_order:
            type_order.append(annotation_type)
    rows: list[tuple[str, str]] = []
    for dataset in dataset_order:
        local_types = [
            item
            for item in type_order
            if (
                ordinary["dataset"].eq(dataset)
                & ordinary["annotation_type"].eq(item)
            ).any()
        ]
        rows.extend((dataset, item) for item in local_types)
    multiplicities = [2, 5, 10]
    matrix = np.full((len(rows), len(multiplicities)), np.nan)
    status = np.full(matrix.shape, "", dtype=object)
    for row_index, (dataset, annotation_type) in enumerate(rows):
        for col_index, multiplicity in enumerate(multiplicities):
            local = ordinary[
                ordinary["dataset"].eq(dataset)
                & ordinary["annotation_type"].eq(annotation_type)
                & ordinary["multiplicity"].eq(multiplicity)
            ]
            if len(local) != 1:
                raise AssertionError(
                    f"Expected one stress cell for {dataset}/{annotation_type}/{multiplicity}"
                )
            value = float(local["signed_response"].iloc[0])
            status[row_index, col_index] = str(local["run_status"].iloc[0])
            if np.isfinite(value):
                matrix[row_index, col_index] = value

    fig, ax = plt.subplots(figsize=(5.8, 4.35))
    cmap = mpl.colormaps["RdBu"].reversed().copy()
    cmap.set_bad("#BFC3C8")
    image = ax.imshow(
        np.ma.masked_invalid(matrix),
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=-0.20, vcenter=0, vmax=0.20),
        aspect="auto",
    )
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            if np.isfinite(value):
                rgba = cmap(TwoSlopeNorm(vmin=-0.20, vcenter=0, vmax=0.20)(value))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                color = "black" if luminance > 0.58 else "white"
                text = f"{value:+.3f}".replace("+0.000", "0.000").replace("-0.000", "0.000")
            else:
                color = DARK
                text = "NF"
            ax.text(col_index, row_index, text, ha="center", va="center", color=color, fontsize=7.0, fontweight="bold")
    labels = [f"{DISPLAY[dataset]} · {annotation_type.replace('_', ' ').title()}" for dataset, annotation_type in rows]
    ax.set_yticks(np.arange(len(rows)), labels)
    ax.set_xticks(np.arange(len(multiplicities)), multiplicities)
    ax.set_xlabel("Total rows per selected weak statement")
    ax.set_ylabel("Materialized annotation type")
    for boundary in range(1, len(rows)):
        if rows[boundary][0] != rows[boundary - 1][0]:
            ax.axhline(boundary - 0.5, color="#8B9198", linewidth=0.8)
    colorbar = fig.colorbar(image, ax=ax, pad=0.025, fraction=0.05)
    colorbar.set_label("Ordinary segment-F1 response")
    colorbar.set_ticks([-0.20, -0.10, 0, 0.10, 0.20])
    fig.subplots_adjust(left=0.31, right=0.88, top=0.98, bottom=0.13)
    save_figure(fig, "type_multiplicity_heatmap.pdf")


def stage_responses() -> None:
    data = pd.read_csv(MATERIALIZATION)
    ordinary = data[
        data["role"].eq("ordinary")
        & data["transformation"].ne("clean")
    ].copy()
    ordinary["configuration"] = (
        ordinary["dataset"].map(DISPLAY)
        + "–"
        + ordinary["architecture"].map(architecture_display)
    )
    ordinary["condition"] = ordinary["configuration"] + " · " + ordinary["transformation"].map(TRANSFORMATION_LABELS).str.replace("\n", " ", regex=False)
    fields = [
        ("maximum_scaler_response", "Scaler"),
        ("maximum_absolute_train_objective_response", "Train objective"),
        ("maximum_absolute_validation_objective_response", "Validation objective"),
        ("maximum_absolute_gradient_response", "Gradient"),
        ("checkpoints_changed", "Checkpoint"),
        ("maximum_probability_response", "Probability"),
        ("decoded_bins_changed", "Decoded bins"),
        ("segment_f1_response_mean", "F1"),
    ]
    ordinary = ordinary.reset_index(drop=True)
    matrix = np.zeros((len(ordinary), len(fields)), dtype=int)
    for row_index, row in ordinary.iterrows():
        for col_index, (field, _) in enumerate(fields):
            value = float(row[field])
            if value == 0:
                matrix[row_index, col_index] = 0
            elif field not in {"checkpoints_changed", "decoded_bins_changed"} and abs(value) < 1e-3:
                matrix[row_index, col_index] = 1
            else:
                matrix[row_index, col_index] = 2
    fig, ax = plt.subplots(figsize=(7.2, 6.45))
    cmap = mpl.colors.ListedColormap(["#F4F5F6", "#C9D4DE", BLUE])
    ax.imshow(matrix, cmap=cmap, vmin=-0.5, vmax=2.5, aspect="auto")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            field = fields[col][0]
            value = float(ordinary.iloc[row][field])
            if field == "checkpoints_changed":
                text = f"{int(value)}/{int(ordinary.iloc[row]['folds'])}"
            elif field == "decoded_bins_changed":
                text = f"{int(value)}"
            else:
                text = {0: "0", 1: "ε", 2: "●"}[matrix[row, col]]
            ax.text(
                col,
                row,
                text,
                ha="center",
                va="center",
                fontsize=6.2,
                color="white" if matrix[row, col] == 2 else DARK,
                fontweight="bold" if field == "checkpoints_changed" else "normal",
            )
    ax.set_xticks(np.arange(len(fields)), [label for _, label in fields], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(ordinary)), ordinary["condition"])
    ax.text(
        1.0,
        1.092,
        "Quotient learning: zero response at every stage shown",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.0,
        color=TEAL,
        fontweight="bold",
    )
    category_handles = [
        Rectangle((0, 0), 1, 1, facecolor="#F4F5F6", edgecolor="#B7BDC5", label="Zero / unchanged"),
        Rectangle((0, 0), 1, 1, facecolor="#C9D4DE", edgecolor="#AAB5C0", label=r"$0<|\Delta|<10^{-3}$"),
        Rectangle((0, 0), 1, 1, facecolor=BLUE, edgecolor=BLUE, label=r"$|\Delta|\geq10^{-3}$ or discrete state changed"),
    ]
    ax.legend(
        handles=category_handles,
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.012),
        columnspacing=1.0,
        handlelength=1.2,
    )
    ax.tick_params(length=0)
    fig.subplots_adjust(left=0.39, right=0.99, top=0.84, bottom=0.08)
    save_figure(fig, "audit_propagation_matrix.pdf")

def optional_iou_counterexample() -> None:
    row = pd.read_csv(IOU_COUNTEREXAMPLE).iloc[0]
    matrix = np.asarray(ast.literal_eval(row["iou_matrix"]), dtype=float)
    truth = ast.literal_eval(row["truth_intervals"])
    prediction = ast.literal_eval(row["prediction_intervals"])
    matched_truth = ast.literal_eval(row["matched_truth_indices"])
    matched_prediction = ast.literal_eval(row["matched_prediction_indices"])
    qualified_pairs = list(zip(matched_truth, matched_prediction, strict=True))
    total_iou_pairs = [(index, index) for index in range(matrix.shape[0])]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), sharey=True)
    for panel_index, (ax, pairs, edgecolor, heading) in enumerate(
        (
            (axes[0], total_iou_pairs, ORANGE, "A  Total-IoU assignment\nbefore qualification"),
            (axes[1], qualified_pairs, TEAL, "B  Qualified cardinality-first\nassignment"),
        )
    ):
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=0.5)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color="white" if matrix[i, j] > 0.35 else DARK,
                    fontweight="bold" if (i, j) in pairs else "normal",
                )
        for i, j in pairs:
            ax.add_patch(
                Rectangle(
                    (j - 0.47, i - 0.47),
                    0.94,
                    0.94,
                    fill=False,
                    edgecolor=edgecolor,
                    linewidth=2.4,
                )
            )
        ax.set_xticks(np.arange(len(prediction)), [str(item) for item in prediction])
        ax.set_xlabel("Predicted interval")
        ax.text(0.0, 1.035, heading, transform=ax.transAxes, ha="left", va="bottom", fontweight="bold", fontsize=7.8)
        if panel_index == 0:
            ax.set_yticks(np.arange(len(truth)), [str(item) for item in truth])
            ax.set_ylabel("Reference interval")
            note = "0.250 + 0.292 + 0.043 = 0.585\n→ 0 qualified matches after the 0.5 threshold"
        else:
            note = "One match at IoU = 0.500\n→ 1 true positive"
        ax.text(
            0.5,
            -0.25,
            note,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7.2,
            color=edgecolor,
            fontweight="bold",
        )
    colorbar_axis = fig.add_axes([0.865, 0.22, 0.022, 0.68])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Temporal IoU")
    fig.subplots_adjust(left=0.14, right=0.82, top=0.82, bottom=0.25, wspace=0.34)
    save_figure(fig, "iou_matching_counterexample.pdf")

def main() -> None:
    validate_inputs()
    compiler_overview()
    core_response()
    transmission_pathways()
    materialization_benchmark()
    evidence_class_composition()
    temporal_examples()
    type_multiplicity_heatmap()
    stage_responses()
    print("Built eight figures from stored results.")

if __name__ == "__main__":
    main()
