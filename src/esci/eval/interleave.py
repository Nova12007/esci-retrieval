"""Team-draft interleaving with simulated cascade clicks.

A methodology demonstration, not evidence about users: the clicks come from a
model whose probabilities are assumptions derived from ESCI labels.

Team-draft (Radlinski, Kurup and Joachims, 2008): two rankers take turns
contributing their best not-yet-shown item, the team with fewer picks going
first and a coin deciding ties. Every user sees one merged list, and each
click credits the team that placed the clicked item, so each session is a
paired comparison. That is why interleaving needs far less traffic than A/B.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

# Assumed click and satisfaction probabilities by ESCI label. None means unjudged.
CLICK = {"E": 0.7, "S": 0.25, "C": 0.1, "I": 0.02, None: 0.05}
STOP_AFTER_CLICK = {"E": 0.7, "S": 0.2, "C": 0.1, "I": 0.0, None: 0.1}


def team_draft(a: list[str], b: list[str], k: int, rng: random.Random) -> list[tuple[str, str]]:
    """Merged list of (product_id, team) with team in {"a", "b"}, length <= k."""
    merged: list[tuple[str, str]] = []
    shown: set[str] = set()
    picks = {"a": 0, "b": 0}
    source = {"a": a, "b": b}
    cursor = {"a": 0, "b": 0}

    def next_unshown(team: str) -> str | None:
        ranking = source[team]
        while cursor[team] < len(ranking) and ranking[cursor[team]] in shown:
            cursor[team] += 1
        return ranking[cursor[team]] if cursor[team] < len(ranking) else None

    while len(merged) < k:
        if picks["a"] != picks["b"]:
            team = "a" if picks["a"] < picks["b"] else "b"
        else:
            team = "a" if rng.random() < 0.5 else "b"
        pid = next_unshown(team)
        if pid is None:  # this team is exhausted; let the other one fill
            team = "b" if team == "a" else "a"
            pid = next_unshown(team)
            if pid is None:
                break
        merged.append((pid, team))
        shown.add(pid)
        picks[team] += 1
    return merged


def cascade_clicks(
    merged: list[tuple[str, str]], labels: dict[str, str], rng: random.Random
) -> dict[str, int]:
    """Scan top-down; click with CLICK[label]; after a click, stop with
    STOP_AFTER_CLICK[label]. Returns clicks per team."""
    clicks = {"a": 0, "b": 0}
    for pid, team in merged:
        label = labels.get(pid)
        if rng.random() < CLICK[label]:
            clicks[team] += 1
            if rng.random() < STOP_AFTER_CLICK[label]:
                break
    return clicks


@dataclass(frozen=True)
class InterleavingResult:
    wins_a: int
    wins_b: int
    ties: int

    @property
    def preference_b(self) -> float:
        """(wins_b - wins_a) / decided sessions, in [-1, 1]; > 0 favours b."""
        decided = self.wins_a + self.wins_b
        return (self.wins_b - self.wins_a) / decided if decided else 0.0


def interleave(
    run_a: dict[int, list[str]],
    run_b: dict[int, list[str]],
    labels: dict[int, dict[str, str]],
    k: int = 10,
    sessions_per_query: int = 1,
    seed: int = 0,
) -> InterleavingResult:
    """Simulate sessions over every query both runs answered."""
    rng = random.Random(seed)
    wins_a = wins_b = ties = 0
    for q in sorted(set(run_a) & set(run_b)):
        for _ in range(sessions_per_query):
            merged = team_draft(run_a[q], run_b[q], k, rng)
            clicks = cascade_clicks(merged, labels.get(q, {}), rng)
            if clicks["a"] > clicks["b"]:
                wins_a += 1
            elif clicks["b"] > clicks["a"]:
                wins_b += 1
            else:
                ties += 1
    return InterleavingResult(wins_a, wins_b, ties)
