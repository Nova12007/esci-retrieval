from __future__ import annotations

import random

from esci.eval.interleave import InterleavingResult, interleave, team_draft


def test_team_draft_alternates_and_never_repeats() -> None:
    merged = team_draft(["a1", "a2", "a3"], ["b1", "b2", "b3"], k=6, rng=random.Random(0))
    pids = [p for p, _ in merged]
    teams = [t for _, t in merged]
    assert len(pids) == len(set(pids)) == 6
    assert teams.count("a") == teams.count("b") == 3
    # each pair of picks has one from each team
    for i in range(0, 6, 2):
        assert {teams[i], teams[i + 1]} == {"a", "b"}


def test_shared_items_are_credited_once_to_whoever_placed_them() -> None:
    merged = team_draft(["x", "a2"], ["x", "b2"], k=3, rng=random.Random(1))
    assert [p for p, _ in merged].count("x") == 1
    assert len(merged) == 3


def test_exhausted_ranker_lets_the_other_fill() -> None:
    merged = team_draft(["a1"], ["b1", "b2", "b3"], k=4, rng=random.Random(2))
    assert [p for p, _ in merged if _ == "b"] == ["b1", "b2", "b3"]


def test_better_ranker_wins_clearly_under_simulated_clicks() -> None:
    labels = {q: {f"good{i}": "E" for i in range(10)} for q in range(300)}
    good = {q: [f"good{i}" for i in range(10)] for q in range(300)}
    bad = {q: [f"junk{i}" for i in range(10)] for q in range(300)}
    result = interleave(bad, good, labels, seed=3)
    assert result.preference_b > 0.9


def test_identical_rankers_are_a_wash() -> None:
    labels = {q: {f"p{i}": "E" for i in range(10)} for q in range(2000)}
    run = {q: [f"p{i}" for i in range(10)] for q in range(2000)}
    assert abs(interleave(run, run, labels, seed=4).preference_b) < 0.08


def test_preference_with_no_decided_sessions_is_zero() -> None:
    assert InterleavingResult(0, 0, 5).preference_b == 0.0
