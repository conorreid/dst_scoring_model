"""D/ST streaming model: rank all defenses for a given NFL week.

Run:  python dst_scoring_model.py [--week N] [--season YYYY] [--csv]
Import:  from dst_scoring_model import run  ->  DataFrame of expected events
         per team plus a 'final' standard-scoring projection.
"""
import argparse
import logging
from datetime import datetime

import pandas as pd

import dst_scoring_model.clean_data as clean_data
import dst_scoring_model.get_tr_data as get_tr_data
import dst_scoring_model.maps as maps
import dst_scoring_model.model as model
import dst_scoring_model.odds as odds


def prior_season_date(season: int) -> str:
    """A date whose teamrankings table reflects the previous season's finals
    (regular season ends very early in January of season+0's calendar +1;
    mid-January of the season year itself safely captures season-1)."""
    return f"{season}-01-10"


def run(week: int | None = None, season: int | None = None,
        scoring: model.Scoring | None = None) -> pd.DataFrame:
    """Build the full ranking table of expected D/ST events for one week.

    Returns a DataFrame with team_name, abbrev, opponent, points_allowed
    (Vegas-implied), sacks, interceptions, fumbles, defensive_touchdowns
    (all expected counts), and 'final' scored with `scoring` (standard if
    omitted). Teams on bye are absent.
    """
    scoring = scoring or model.Scoring()
    spreads = odds.get_lines(week, season)
    if spreads.empty:
        raise RuntimeError("no games/lines returned for that week")
    season = season or datetime.now().year
    week = week or 1
    prior = prior_season_date(season)

    tr_items = {}
    for key, value in maps.tr_stat_list.items():
        tr_items[key] = get_tr_data.get_tr_stats_full(
            value["url"], value["column_name"], week, prior)

    logging.info("merging defense-side stats")
    defense_df = pd.merge(spreads, tr_items["sacks_defense_list"],
                          how="left", on="team_name")
    for item in maps.defense_pure_list:
        defense_df = pd.merge(defense_df, tr_items[item], how="left",
                              on="team_name")

    logging.info("merging opponent-side stats")
    opponent_df = pd.merge(tr_items["sacks_offense_list"],
                           tr_items["interceptions_thrown"],
                           how="left", on="team_name")
    opponent_df = pd.merge(opponent_df, tr_items["fumbles_thrown"],
                           how="left", on="team_name")

    fused_df = pd.merge(defense_df, opponent_df, how="left",
                        left_on="opponent", right_on="team_name",
                        suffixes=("", "_opp"))

    for key, cap in maps.poisson_events.items():
        fused_df[key] = fused_df[key].apply(
            lambda rate: model.poisson_create(rate, cap))
    for item in maps.defense_opponent_list:
        fused_df = clean_data.defense_opponent_fusion(fused_df, item)

    fused_df["final"] = fused_df.apply(scoring.score, axis=1)
    fused_df["abbrev"] = fused_df["team_name"].map(maps.team_to_abbrev)
    fused_df = fused_df[["team_name", "abbrev", "final", "opponent",
                         "points_allowed", "interceptions", "fumbles",
                         "sacks", "defensive_touchdowns"]]
    return fused_df.sort_values(by="final", ascending=False,
                                ignore_index=True)
