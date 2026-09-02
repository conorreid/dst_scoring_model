"""Poisson event model and configurable fantasy scoring."""
from dataclasses import dataclass, field

import numpy as np
import scipy.stats as stats


def poisson_create(rate: float, max_possible: int) -> float:
    """Expected value of a Poisson-distributed event truncated at max_possible.

    Args:
        rate: average per-game rate of the event
        max_possible: cap on occurrences considered
    Returns:
        sum over k of k * P(k) for k in [0, max_possible)
    """
    n = np.arange(0, max_possible)
    return float((n * stats.poisson.pmf(n, rate)).sum())


# (upper_exclusive_bound, score): standard ESPN-style points-allowed brackets
STANDARD_PA_BRACKETS = ((1, 10), (7, 7), (14, 4), (18, 1), (28, 0),
                        (35, -3), (10 ** 9, -4))


@dataclass
class Scoring:
    """Per-league D/ST scoring: event point values + points-allowed brackets."""
    sack: float = 1.0
    interception: float = 2.0
    fumble: float = 2.0
    defensive_touchdown: float = 6.0
    pa_brackets: tuple = field(default_factory=lambda: STANDARD_PA_BRACKETS)

    def points_allowed_score(self, points: float) -> float:
        for bound, score in self.pa_brackets:
            if points < bound:
                return score
        return self.pa_brackets[-1][1]

    def expected_pa_score(self, implied_points: float, max_points: int = 60) -> float:
        """E[bracket score] over a Poisson distribution of points allowed.

        The brackets are nonlinear, so scoring the mean understates variance
        effects; taking the expectation over the full distribution is the
        statistically correct version (the old model scored the mean).
        """
        ks = np.arange(0, max_points)
        pmf = stats.poisson.pmf(ks, implied_points)
        scores = np.array([self.points_allowed_score(k) for k in ks])
        return float((pmf * scores).sum())

    def score(self, row) -> float:
        """Total expected fantasy points for one team's expected-event row."""
        return (self.expected_pa_score(row["points_allowed"])
                + row["interceptions"] * self.interception
                + row["sacks"] * self.sack
                + row["fumbles"] * self.fumble
                + row["defensive_touchdowns"] * self.defensive_touchdown)


def scoring_from_sleeper(settings: dict) -> Scoring:
    """Build a Scoring config from a Sleeper league's scoring_settings dict."""
    brackets = ((1, settings.get("pts_allow_0", 10)),
                (7, settings.get("pts_allow_1_6", 7)),
                (14, settings.get("pts_allow_7_13", 4)),
                (21, settings.get("pts_allow_14_20", 1)),
                (28, settings.get("pts_allow_21_27", 0)),
                (35, settings.get("pts_allow_28_34", -1)),
                (10 ** 9, settings.get("pts_allow_35p", -4)))
    return Scoring(sack=settings.get("sack", 1.0),
                   interception=settings.get("int", 2.0),
                   fumble=settings.get("fum_rec", 2.0),
                   defensive_touchdown=settings.get("def_td", 6.0),
                   pa_brackets=brackets)


def points_allowed_score(points: float) -> int:
    """Standard-scoring bracket lookup (kept for backwards compatibility)."""
    return int(Scoring().points_allowed_score(points))
