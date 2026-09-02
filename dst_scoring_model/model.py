"""Poisson event model, fitted parameters, and configurable fantasy scoring."""
import json
import os
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
import scipy.stats as stats

PARAMS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "fitted_params.json")

POISSON_CAPS = {"sacks": 10, "interceptions": 7, "fumbles": 6,
                "defensive_touchdowns": 4}


def poisson_create(rate: float, max_possible: int) -> float:
    """Expected value of a Poisson-distributed event truncated at max_possible."""
    n = np.arange(0, max_possible)
    return float((n * stats.poisson.pmf(n, rate)).sum())


@dataclass
class ModelParams:
    """Fusion weights (share on the defense's own rate vs the opponent's
    giveaway rate), game-script sensitivity, and fumble recovery rate.
    Defaults are the backtest-fitted values when fitted_params.json exists."""
    w_sacks: float = 0.45
    w_ints: float = 0.45
    w_fumbles: float = 0.45
    game_script_k: float = 0.0   # pass-pressure multiplier per implied point of margin
    fumble_recovery_rate: float = 0.5

    @classmethod
    def fitted(cls) -> "ModelParams":
        try:
            return cls(**json.load(open(PARAMS_FILE)))
        except Exception:
            return cls()

    def save(self):
        json.dump(asdict(self), open(PARAMS_FILE, "w"), indent=2)


def predict(rate_df: pd.DataFrame, lines_df: pd.DataFrame,
            params: ModelParams | None = None) -> pd.DataFrame:
    """Expected D/ST events for each team with a game this week.

    Args:
        rate_df: per-game rates indexed by team abbrev (rates.rates_asof)
        lines_df: one row per team: team, opponent, points_allowed (implied),
            margin (implied points for minus implied points allowed)
        params: fitted model parameters (loads fitted_params.json if omitted)
    Returns:
        DataFrame: team, opponent, points_allowed, sacks, interceptions,
        fumbles (expected recoveries), defensive_touchdowns (incl. returns)
    """
    p = params or ModelParams.fitted()
    rows = []
    for line in lines_df.itertuples():
        if line.team not in rate_df.index or line.opponent not in rate_df.index:
            continue
        d = rate_df.loc[line.team]      # this defense's own rates
        o = rate_df.loc[line.opponent]  # opponent giveaway rates
        # game script: a favored defense faces more dropbacks — scale the
        # pass-pressure events (sacks, interceptions) by implied margin
        script = float(np.clip(1 + p.game_script_k * line.margin, 0.75, 1.25))
        sacks = (p.w_sacks * d["def_sacks"]
                 + (1 - p.w_sacks) * o["sacks_suffered"]) * script
        ints = (p.w_ints * d["def_interceptions"]
                + (1 - p.w_ints) * o["passing_interceptions"]) * script
        # turnover-luck regression: fuse fumble *events* (forced / put on the
        # ground), then apply a league recovery rate instead of trusting
        # noisy historical recovery counts
        fumble_events = (p.w_fumbles * d["def_fumbles_forced"]
                         + (1 - p.w_fumbles) * o["off_fumbles"])
        fumbles = fumble_events * p.fumble_recovery_rate
        tds = d["def_tds"] + d["special_teams_tds"]
        rows.append({
            "team": line.team, "opponent": line.opponent,
            "points_allowed": line.points_allowed,
            "sacks": poisson_create(sacks, POISSON_CAPS["sacks"]),
            "interceptions": poisson_create(ints, POISSON_CAPS["interceptions"]),
            "fumbles": poisson_create(fumbles, POISSON_CAPS["fumbles"]),
            "defensive_touchdowns": poisson_create(
                tds, POISSON_CAPS["defensive_touchdowns"]),
        })
    return pd.DataFrame(data=rows)


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
        effects; the expectation over the full distribution is correct.
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

    def score_actual(self, pa, sacks, ints, fumbles, tds) -> float:
        """Fantasy points for realized (integer) stat totals."""
        return (self.points_allowed_score(pa) + sacks * self.sack
                + ints * self.interception + fumbles * self.fumble
                + tds * self.defensive_touchdown)


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
