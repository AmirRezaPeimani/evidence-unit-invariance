#!/usr/bin/env python3
"""Validate and print the stored operational-materialization summary."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "materialization_benchmark.csv"


def main() -> None:
    data = pd.read_csv(RESULT)
    ordinary = data[
        data["role"].eq("ordinary")
        & data["transformation"].ne("clean")
    ].copy()
    quotient = data[
        data["role"].eq("quotient")
        & data["transformation"].ne("clean")
    ].copy()
    if len(ordinary) != 18:
        raise AssertionError(f"Expected 18 ordinary conditions, found {len(ordinary)}")

    f1_changed = int(ordinary["segment_f1_response_mean"].round(3).ne(0).sum())
    probability_changed = int(
        ordinary["maximum_probability_response"].round(3).ne(0).sum()
    )
    both_null = int(
        (
            ordinary["segment_f1_response_mean"].round(3).eq(0)
            & ordinary["maximum_probability_response"].round(3).eq(0)
        ).sum()
    )
    reorder = ordinary[
        ordinary["transformation"].eq("deterministic_row_reorder")
    ]
    maximum_probability = float(ordinary["maximum_probability_response"].max())

    assert (f1_changed, probability_changed, both_null) == (12, 13, 5)
    assert reorder["segment_f1_response_mean"].round(3).eq(0).all()
    assert reorder["maximum_probability_response"].round(3).eq(0).all()
    assert round(maximum_probability, 3) == 0.587
    assert quotient["segment_f1_response_mean"].eq(0).all()
    assert quotient["maximum_probability_response"].eq(0).all()

    print(f"ordinary F1 changes: {f1_changed}/18")
    print(f"ordinary probability changes: {probability_changed}/18")
    print(f"null in both at three decimals: {both_null}/18")
    print(
        "row reorder: zero segment-F1 response and sub-1e-3 ordinary "
        "probability responses; zero at three-decimal precision"
    )
    print(f"maximum absolute probability response: {maximum_probability:.3f}")
    print("quotient responses: zero in reported conditions")


if __name__ == "__main__":
    main()
