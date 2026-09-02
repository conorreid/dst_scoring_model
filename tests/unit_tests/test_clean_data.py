import pandas as pd

import dst_scoring_model.clean_data as clean_data


def test_defense_opponent_fusion():
    df = pd.DataFrame(data={"stat_thrown": [1.0], "stat_created": [2.0]})
    out = clean_data.defense_opponent_fusion(df, "stat")
    assert abs(out["stat"][0] - 1.45) < 1e-9
    assert "stat_thrown" not in out.columns
