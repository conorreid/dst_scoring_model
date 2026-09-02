"""nflverse data loaders: weekly team stats and schedules with Vegas lines.

Both the live model and the backtest build their stat rates from these files,
so backtest results validate the exact code path that runs in-season.
"""
import logging
import os
import time

import pandas as pd

CACHE_DIR = os.path.expanduser("~/.cache/dst_scoring_model")
STATS_URL = ("https://github.com/nflverse/nflverse-data/releases/download/"
             "stats_team/stats_team_week_{season}.csv")
GAMES_URL = "http://www.habitatring.com/games.csv"

# nflverse team codes that differ from ours
NFLVERSE_TO_ABBREV = {"LA": "LAR", "WAS": "WSH", "OAK": "LV", "SD": "LAC"}

STAT_COLS = ["def_sacks", "def_interceptions", "def_fumbles_forced",
             "fumble_recovery_opp", "def_tds", "special_teams_tds",
             "sacks_suffered", "passing_interceptions", "off_fumbles",
             "attempts"]


def _cached_csv(url: str, name: str, ttl: float | None) -> pd.DataFrame:
    """Download a CSV with local caching. ttl=None means cache forever."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, name)
    stale = (not os.path.exists(path)
             or (ttl is not None and time.time() - os.path.getmtime(path) > ttl))
    if stale:
        logging.info("downloading %s", url)
        df = pd.read_csv(url)
        df.to_csv(path, index=False)
        return df
    return pd.read_csv(path)


def team_week_stats(season: int, final: bool = True) -> pd.DataFrame:
    """Per-team per-week stats for one season (regular season only).

    final=False refreshes every 6h (use for the in-progress season).
    Raises on seasons nflverse hasn't published yet (e.g. before week 1).
    """
    df = _cached_csv(STATS_URL.format(season=season), f"stats_{season}.csv",
                     None if final else 6 * 3600)
    df = df[df["season_type"] == "REG"].copy()
    for col in ("team", "opponent_team"):
        df[col] = df[col].replace(NFLVERSE_TO_ABBREV)
    df["off_fumbles"] = (df["rushing_fumbles"] + df["receiving_fumbles"]
                         + df["sack_fumbles"])
    return df


def games(ttl: float = 6 * 3600) -> pd.DataFrame:
    """Schedules with scores and closing Vegas lines (spread_line is
    home-relative, positive = home favored)."""
    df = _cached_csv(GAMES_URL, "games.csv", ttl)
    df = df[df["game_type"] == "REG"].copy()
    for col in ("home_team", "away_team"):
        df[col] = df[col].replace(NFLVERSE_TO_ABBREV)
    return df
