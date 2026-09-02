"""Per-game stat rates scraped from teamrankings.com.

Each stat page serves one table with season-to-date, last-3-games, and other
splits. A ?date=YYYY-MM-DD query serves the table as of that date, which is how
the previous season's full-year rates are fetched as a prior.
"""
import logging

import numpy as np
import pandas as pd
import requests

import dst_scoring_model.maps as maps

UA = {"User-Agent": "Mozilla/5.0 (dst_scoring_model)"}


def get_tr_stats(url: str, stat_name: str) -> pd.DataFrame:
    """Scrape one teamrankings stat table into a dataframe.

    Args:
        url: stat page URL (optionally with ?date=)
        stat_name: prefix for the produced columns
    Returns:
        DataFrame with team_name, <stat>_season, <stat>_last_3
    """
    from bs4 import BeautifulSoup

    logging.info("getting teamrankings stats for %s from %s", stat_name, url)
    html = BeautifulSoup(requests.get(url, headers=UA, timeout=30).text,
                         "html.parser")
    table = html.find("table", class_="tr-table datatable scrollable")
    rows = []
    for tr in table.find_all("tr")[1:]:
        tds = tr.find_all("td")

        def num(td):
            try:
                return float(td.text)
            except ValueError:
                return np.nan

        rows.append({"team_name": tds[1].text,
                     stat_name + "_season": num(tds[2]),
                     stat_name + "_last_3": num(tds[3])})
    df = pd.DataFrame(data=rows)
    df["team_name"] = df["team_name"].map(maps.town_to_team)
    return df


def blend_weight(week: int) -> float:
    """Weight on current-season rates, ramping 0 -> 1 over the first ~10 weeks.

    Week 1 uses the previous season entirely; by week 11 the prior is gone.
    """
    return min(max(week - 1, 0), 10) / 10


def get_tr_stats_full(url: str, stat_name: str, week: int,
                      prior_date: str) -> pd.DataFrame:
    """Fetch current + prior-season rates for one stat and blend by week.

    Current-season signal is itself a mix of season-to-date (0.5) and the
    last 3 games (0.5) to capture recency.

    Args:
        url: teamrankings stat page URL
        stat_name: output column name
        week: current NFL week (drives the prior/current blend)
        prior_date: YYYY-MM-DD date whose table reflects last season's finals
    Returns:
        DataFrame with team_name, <stat_name>
    """
    prior = get_tr_stats(f"{url}?date={prior_date}", stat_name + "_prior")
    w = blend_weight(week)
    if w == 0:
        df = prior
        df[stat_name] = df[stat_name + "_prior_season"]
    else:
        cur = get_tr_stats(url, stat_name + "_cur")
        df = pd.merge(prior, cur, how="left", on="team_name")
        current = (df[stat_name + "_cur_season"].fillna(0) * 0.5
                   + df[stat_name + "_cur_last_3"].fillna(0) * 0.5)
        df[stat_name] = (1 - w) * df[stat_name + "_prior_season"] + w * current
    return df[["team_name", stat_name]]
