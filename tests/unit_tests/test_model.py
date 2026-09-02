import dst_scoring_model.model as model


def test_poisson_create():
    # expected value of truncated Poisson approaches the rate
    assert abs(model.poisson_create(2.5, 20) - 2.5) < 0.01


def test_points_allowed_score_brackets():
    s = model.Scoring()
    assert s.points_allowed_score(0) == 10
    assert s.points_allowed_score(6) == 7
    assert s.points_allowed_score(13) == 4
    assert s.points_allowed_score(17) == 1
    assert s.points_allowed_score(27) == 0
    assert s.points_allowed_score(34) == -3
    assert s.points_allowed_score(50) == -4


def test_expected_pa_score_below_bracket_of_mean():
    # variance spreads mass into worse brackets around a 21-point mean
    s = model.Scoring()
    assert s.expected_pa_score(21) < s.points_allowed_score(21) + 1


def test_scoring_from_sleeper():
    s = model.scoring_from_sleeper(
        {"sack": 1.5, "int": 2.0, "fum_rec": 2.0, "def_td": 6.0,
         "pts_allow_0": 12.0})
    assert s.sack == 1.5
    assert s.points_allowed_score(0) == 12.0


def test_score_row():
    s = model.Scoring()
    row = {"points_allowed": 17.0, "interceptions": 1.0, "sacks": 3.0,
           "fumbles": 0.5, "defensive_touchdowns": 0.1}
    expected = (s.expected_pa_score(17.0) + 2 + 3 + 1 + 0.6)
    assert abs(s.score(row) - expected) < 1e-9
