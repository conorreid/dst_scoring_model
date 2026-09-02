import unittest.mock as mock

import dst_scoring_model.odds as odds


FAKE = {"events": [{"competitions": [{
    "competitors": [
        {"homeAway": "home", "team": {"displayName": "Seattle Seahawks"}},
        {"homeAway": "away", "team": {"displayName": "New England Patriots"}},
    ],
    "odds": [{"overUnder": 44.5, "spread": -3.5}],
}]}]}


@mock.patch("requests.get")
def test_get_lines(get):
    get.return_value.json.return_value = FAKE
    df = odds.get_lines(1, 2026)
    home = df[df.team_name == "Seattle Seahawks"].iloc[0]
    away = df[df.team_name == "New England Patriots"].iloc[0]
    # SEA favored by 3.5, total 44.5 -> SEA allows 20.5, NE allows 24
    assert home.points_allowed == 20.5
    assert away.points_allowed == 24.0
    assert home.opponent == "New England Patriots"
