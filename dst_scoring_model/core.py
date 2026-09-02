"""Live D/ST rankings: current Vegas lines + nflverse rates + fitted params."""
import logging
from datetime import datetime

import pandas as pd

import dst_scoring_model.maps as maps
import dst_scoring_model.model as model
import dst_scoring_model.odds as odds
import dst_scoring_model.rates as rates


def run(week: int | None = None, season: int | None = None,
        scoring: model.Scoring | None = None,
        params: model.ModelParams | None = None) -> pd.DataFrame:
    """Rank all defenses with a game this week.

    Returns a DataFrame with team_name, abbrev, final (scored with `scoring`,
    standard if omitted), opponent, points_allowed (Vegas-implied), and
    expected sacks, interceptions, fumbles (recoveries), and
    defensive_touchdowns (incl. return TDs). Teams on bye are absent.
    """
    scoring = scoring or model.Scoring()
    lines = odds.get_lines(week, season)
    if lines.empty:
        raise RuntimeError("no games/lines returned for that week")
    season = season or datetime.now().year
    week = week or 1

    abbrev = maps.team_to_abbrev
    lines["team"] = lines["team_name"].map(abbrev)
    lines["opp_abbrev"] = lines["opponent"].map(abbrev)
    lines["margin"] = lines["points_for"] - lines["points_allowed"]
    rate_df = rates.rates_asof(season, week)

    pred = model.predict(
        rate_df,
        lines.rename(columns={"opponent": "opponent_name"})
             .rename(columns={"opp_abbrev": "opponent"}),
        params)
    logging.info("predicted %d defenses", len(pred))
    to_name = {v: k for k, v in abbrev.items()}
    pred["team_name"] = pred["team"].map(to_name)
    pred["opponent"] = pred["opponent"].map(to_name)
    pred["abbrev"] = pred["team"]
    pred["final"] = pred.apply(scoring.score, axis=1)
    pred = pred[["team_name", "abbrev", "final", "opponent", "points_allowed",
                 "interceptions", "fumbles", "sacks", "defensive_touchdowns"]]
    return pred.sort_values(by="final", ascending=False, ignore_index=True)
