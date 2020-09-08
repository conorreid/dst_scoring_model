import logging

import pandas as pd


def defense_opponent_fusion(df: pd.DataFrame, stat: str) -> pd.DataFrame:
    """Create the composite number for each stat, fusing offense and defense.

    Args:
        df (pandas.Dataframe): df of the full table
        stat (string): stat to fuse
    Return:s
        df (pandas.Dataframe): df with new/dropped column(s)
    """
    logging.info("fusing stats for " + stat)
    df[stat] = df[stat + "_created"] * 0.45 + df[stat + "_thrown"] * 0.55
    df = df.drop([stat + "_created", stat + "_thrown"], axis=1)
    return df


def interceptions_merge(row: pd.Series) -> float:
    """Takes in the row of merged tr and qb dfs and returns consolidated int #

    This function is made for primarily handling missing QB interception data
    amongst rookies. If the rookie is not in the QB historical interception
    data, this function will return the team's average interceptions thrown in
    a game. If the QB is indeed in the historical interception data, it will
    use that number instead.
    """
    logging.info("checking ints for row: " + str(row))
    if pd.isna(row["interceptions_per_game_qb"]):
        return row["interceptions_thrown"]
    else:
        return row["interceptions_per_game_qb"]
