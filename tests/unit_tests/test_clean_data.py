import pandas as pd

import dst_scoring_model.clean_data as clean_data


def test_defense_opponent_fusion():
    data_for_df = {'stat_thrown': [1, 1], 'stat_created': [2, 2]}
    df_data = pd.DataFrame(data=data_for_df)
    assert clean_data.defense_opponent_fusion(df_data, 'stat')['stat'][0] == 1.4500000000000002