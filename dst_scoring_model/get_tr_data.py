import logging

import dst_scoring_model.maps as maps
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup


def get_tr_stats(url: str, stat_name: str) -> pd.DataFrame:
    """Calls out to the URL specified at teamrankings.com, finds the table using
    BeautifulSoup, then sends table to construct_tr_df to make and return the
    dataframe.

    Args:
        stat_name (string): the stat name in question
        url (string): url of the site to scrape
    Returns:
        tr_df (pandas.Dataframe): dataframe of specified stat per team
    """
    logging.info("getting teamrankings stats for " + stat_name + " for url " + url)
    queried_stat = requests.get(url)
    queried_stat_html = BeautifulSoup(queried_stat.text, "html.parser")
    table = queried_stat_html.find("table", class_="tr-table datatable scrollable")
    queried_stat_list = []
    for tr in table.find_all("tr")[1:]:
        tds = tr.find_all("td")
        try:
            queried_stat_list.append(
                {
                    "team_name": tds[1].text,
                    stat_name + "_season": float(tds[2].text),
                    stat_name + "_last_3": float(tds[3].text),
                }
            )
        except ValueError:
            queried_stat_list.append(
                {
                    "team_name": tds[1].text,
                    stat_name + "_season": np.nan,
                    stat_name + "_last_3": np.nan,
                }
            )
    return construct_tr_df(queried_stat_list)


def get_tr_stats_full(url: str, stat_name: str) -> pd.DataFrame:
    """Calls out to URL for both 2019 and 2020 stats, fuses dfs, and returns
    a dataframe.

    Args:
        url (string): url of the site to scrape
        stat_name (string): the stat name in question
    Returns:
        tr_df (pandas.Dataframe): dataframe of specified stat per team
    """
    df1 = get_tr_stats(url, stat_name + "2019")
    df2 = get_tr_stats(url + "?date=2020-09-05", stat_name + "2020")
    df_merge = pd.merge(df1, df2, how="left", on="team_name", suffixes=("", "_x"))
    df_merge[stat_name] = (
        df_merge[stat_name + "2019_season"] * 0.7
        + df_merge[stat_name + "2020_last_3"] * 0.3
    )
    df_merge = df_merge[["team_name", stat_name]]
    return df_merge


def construct_tr_df(tr_list: list) -> pd.DataFrame:
    """Takes in the list of dictionaries, converts to a dataframe, and then
    maps town_to_team dict as well as computes average of last season and last
    3 games score.

    Args:
        tr_list (list): list of dictionaries from tr_stats()
    Returns:
        tr_df (pandas.Dataframe): dataframe of specified stat per team
    """
    tr_df = pd.DataFrame(data=tr_list)
    tr_df["team_name"] = tr_df["team_name"].map(maps.town_to_team)
    return tr_df
