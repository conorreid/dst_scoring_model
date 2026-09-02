"""Per-team per-game stat rates as of a given (season, week).

Blends the previous season's full-year rates with the current season
(half season-to-date, half last-3-games), ramping from all-prior in week 1
to all-current by week 11. Uses only data from before the target week, so
the same function serves live predictions and leak-free backtesting.
"""
import logging

import pandas as pd

from dst_scoring_model import data


def blend_weight(week: int) -> float:
    """Weight on current-season rates, ramping 0 -> 1 over the first ~10 weeks."""
    return min(max(week - 1, 0), 10) / 10


def per_game(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate team-week rows into per-game rates, indexed by team."""
    grouped = df.groupby("team")[data.STAT_COLS].sum()
    games_played = df.groupby("team").size()
    return grouped.div(games_played, axis=0)


def rates_asof(season: int, week: int) -> pd.DataFrame:
    """Blended per-game rates using only data strictly before (season, week)."""
    prior = per_game(data.team_week_stats(season - 1))
    w = blend_weight(week)
    if w > 0:
        try:
            cur = data.team_week_stats(season, final=False)
            cur = cur[cur["week"] < week]
        except Exception as exc:
            logging.info("current-season stats unavailable (%s); prior only", exc)
            cur = pd.DataFrame()
        if not cur.empty:
            season_rates = per_game(cur)
            last3 = per_game(cur[cur["week"] >= week - 3])
            current = 0.5 * season_rates + 0.5 * last3.reindex(season_rates.index)
            blended = ((1 - w) * prior.reindex(current.index) + w * current)
            return blended.fillna(prior)
    return prior
