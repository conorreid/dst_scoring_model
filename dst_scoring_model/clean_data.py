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
    logging.info('fusing stats for ' + stat)
    df[stat] = df[stat + '_created'] * 0.45 + df[stat + '_thrown'] * 0.55
    df = df.drop([stat + '_created', stat + '_thrown'], axis=1)
    return df