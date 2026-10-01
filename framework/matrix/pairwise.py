"""Deterministic pairwise (2-wise) test selection.

Algorithm: greedy set cover over uncovered value pairs, in the spirit of AETG /
IPOG but simplified. Ties are broken by the stable input order, so the same
input always yields the same output. **No optimality claim is made**; the
selector guarantees that every *feasible* pair is covered and reports the
achieved coverage explicitly instead of asserting it.

Feasible pair
    A (dimension_a=value_a, dimension_b=value_b) pair that appears in at least
    one feasible (non-UNSUPPORTED) combination. Pairs that only occur in
    unsupported combinations are infeasible and are reported separately rather
    than silently dropped.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

from framework.matrix.generator import MatrixGenerator, TestCase

Pair = tuple[tuple[str, str], tuple[str, str]]  # ((dim_a, value_a), (dim_b, value_b))


@dataclass
class PairwiseResult:
    selected: list[TestCase]
    total_candidates: int
    total_pairs_in_full_matrix: int
    feasible_pairs: int
    covered_pairs: int
    infeasible_pairs: list[Pair] = field(default_factory=list)

    @property
    def coverage_percent(self) -> float:
        return round(100.0 * self.covered_pairs / self.feasible_pairs, 1) if self.feasible_pairs else 100.0

    @property
    def reduction_percent(self) -> float:
        if self.total_candidates == 0:
            return 0.0
        return round(100.0 * (1 - len(self.selected) / self.total_candidates), 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": "pairwise",
            "candidates_feasible": self.total_candidates,
            "selected": len(self.selected),
            "reduction_percent": self.reduction_percent,
            "pairs_in_full_matrix": self.total_pairs_in_full_matrix,
            "feasible_pairs": self.feasible_pairs,
            "covered_pairs": self.covered_pairs,
            "pair_coverage_percent": self.coverage_percent,
            "infeasible_pairs": [f"{a[0]}={a[1]} & {b[0]}={b[1]}" for a, b in self.infeasible_pairs],
        }


def pairs_of(case: TestCase, dimensions: tuple[str, ...] = MatrixGenerator.DIMENSIONS) -> set[Pair]:
    values = case.configuration.to_dict()
    return {((a, values[a]), (b, values[b])) for a, b in itertools.combinations(dimensions, 2)}


class PairwiseSelector:
    def __init__(self, dimensions: tuple[str, ...] = MatrixGenerator.DIMENSIONS) -> None:
        self._dimensions = dimensions

    def select(self, cases: list[TestCase]) -> PairwiseResult:
        all_pairs: set[Pair] = set()
        for case in cases:
            all_pairs |= pairs_of(case, self._dimensions)

        candidates = [c for c in cases if c.feasible]
        candidate_pairs = {c.case_id: pairs_of(c, self._dimensions) for c in candidates}
        feasible_pairs: set[Pair] = set().union(*candidate_pairs.values()) if candidate_pairs else set()
        infeasible = sorted(all_pairs - feasible_pairs)

        uncovered = set(feasible_pairs)
        selected: list[TestCase] = []
        remaining = list(candidates)  # stable order = deterministic tie-breaking
        while uncovered and remaining:
            best = max(remaining, key=lambda c: len(candidate_pairs[c.case_id] & uncovered))
            gain = candidate_pairs[best.case_id] & uncovered
            if not gain:
                break
            best.selected = True
            best.selection_reason = f"pairwise: covers {len(gain)} new pair(s)"
            selected.append(best)
            uncovered -= gain
            remaining.remove(best)

        return PairwiseResult(
            selected=selected,
            total_candidates=len(candidates),
            total_pairs_in_full_matrix=len(all_pairs),
            feasible_pairs=len(feasible_pairs),
            covered_pairs=len(feasible_pairs) - len(uncovered),
            infeasible_pairs=infeasible,
        )
