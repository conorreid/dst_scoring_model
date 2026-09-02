"""Vegas lines via ESPN's public scoreboard API (DraftKings odds).

Replaces the old therundown.io/Pinnacle source, which no longer serves data.
"""
import logging

import pandas as pd
import requests

SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"


def get_lines(week: int | None = None, season: int | None = None) -> pd.DataFrame:
    """Fetch spreads/totals and derive each team's implied points allowed.

    The spread is home-relative (negative = home favored), so:
      home implied score = (total - spread) / 2
      away implied score = (total + spread) / 2
    and a team's implied points allowed is its opponent's implied score.

    Args:
        week: NFL week; ESPN defaults to the current week when omitted.
        season: season year; ESPN defaults to the current season when omitted.
    Returns:
        DataFrame with team_name, opponent, points_allowed. Teams on bye are
        absent. Games with no posted line fall back to 22 points allowed each.
    """
    params = {"seasontype": 2}
    if week is not None:
        params["week"] = week
    if season is not None:
        params["dates"] = season
    logging.info("getting lines from ESPN scoreboard %s", params)
    data = requests.get(SCOREBOARD, params=params, timeout=30).json()
    rows = []
    for event in data.get("events", []):
        comp = event["competitions"][0]
        teams = {c["homeAway"]: c["team"]["displayName"] for c in comp["competitors"]}
        odds = (comp.get("odds") or [{}])[0]
        total, spread = odds.get("overUnder"), odds.get("spread")
        if total is not None and spread is not None:
            home_pts = (total - spread) / 2
            away_pts = (total + spread) / 2
        else:
            home_pts = away_pts = 22.0
        rows.append({"team_name": teams["home"], "opponent": teams["away"],
                     "points_allowed": away_pts, "points_for": home_pts})
        rows.append({"team_name": teams["away"], "opponent": teams["home"],
                     "points_allowed": home_pts, "points_for": away_pts})
    return pd.DataFrame(data=rows)
