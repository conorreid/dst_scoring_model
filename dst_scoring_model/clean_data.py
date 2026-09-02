import logging

import pandas as pd


def defense_opponent_fusion(df: pd.DataFrame, stat: str) -> pd.DataFrame:
    """Create the composite number for each stat, fusing what the defense
    creates with what its opponent gives up.

    Args:
        df: full merged table
        stat: stat to fuse (expects <stat>_created and <stat>_thrown columns)
    Returns:
        df with the fused column, source columns dropped
    """
    logging.info("fusing stats for %s", stat)
    df[stat] = df[stat + "_created"] * 0.45 + df[stat + "_thrown"] * 0.55
    df = df.drop([stat + "_created", stat + "_thrown"], axis=1)
    return df
