import pandas as pd

import dst_scoring_model.model as model


def make_rates():
    return pd.DataFrame(
        {"def_sacks": [3.0, 2.0], "sacks_suffered": [2.0, 3.0],
         "def_interceptions": [1.0, 0.5], "passing_interceptions": [0.8, 1.2],
         "def_fumbles_forced": [1.0, 0.6], "off_fumbles": [1.2, 1.4],
         "def_tds": [0.1, 0.05], "special_teams_tds": [0.05, 0.0],
         "fumble_recovery_opp": [0.5, 0.4], "attempts": [33.0, 35.0]},
        index=["AAA", "BBB"])


def test_predict_fusion_and_script():
    p = model.ModelParams(w_sacks=0.5, w_ints=0.5, w_fumbles=1.0,
                          game_script_k=0.01, fumble_recovery_rate=0.4)
    lines = pd.DataFrame({"team": ["AAA"], "opponent": ["BBB"],
                          "points_allowed": [20.0], "margin": [7.0]})
    out = model.predict(make_rates(), lines, p)
    row = out.iloc[0]
    # sacks: (0.5*3 + 0.5*3) * (1 + 0.01*7) = 3.21 (minus tiny truncation)
    assert abs(row["sacks"] - 3.21) < 0.05
    # fumbles: w=1 -> forced rate 1.0 * recovery 0.4
    assert abs(row["fumbles"] - 0.4) < 0.02
    # tds: def 0.1 + st 0.05
    assert abs(row["defensive_touchdowns"] - 0.15) < 0.01


def test_predict_skips_unknown_teams():
    lines = pd.DataFrame({"team": ["AAA"], "opponent": ["ZZZ"],
                          "points_allowed": [20.0], "margin": [0.0]})
    assert model.predict(make_rates(), lines, model.ModelParams()).empty


def test_params_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(model, "PARAMS_FILE", str(tmp_path / "p.json"))
    p = model.ModelParams(w_sacks=0.35, game_script_k=0.005)
    p.save()
    assert model.ModelParams.fitted() == p
