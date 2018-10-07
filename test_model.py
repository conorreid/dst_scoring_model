import model
import pytest
import pandas as pd
import unittest.mock as mock


def construct_request_content():
    response = mock.MagicMock()
    response.content = mock.MagicMock(return_value='hello')
    return response


@pytest.mark.parametrize("test_input, expect", [(0, 10), (6, 7), (13, 4),
                                                (17, 1), (27, 0), (34, -3),
                                                (36, -4)])
def test_points_allowed_score(test_input: int, expect: int) -> bool:
    assert model.points_allowed_score(test_input) == expect


def test_poisson_create():
    assert model.poisson_create(1, 2) == 0.36787944117144233


def test_defense_opponent_fusion():
    data_for_df = {'stat_thrown': [1, 1], 'stat_created': [2, 2]}
    df_data = pd.DataFrame(data=data_for_df)
    assert model.defense_opponent_fusion(df_data, 'stat')['stat'][0] == 1.4500000000000002


@mock.patch('requests.get', return_value=construct_request_content())
@mock.patch('json.loads',
            return_value={'events': [
                {'teams': [
                    {'name': 'hello'}, {'name': 'goodbye'}],
                 'lines': {'3': {'spread': {'point_spread_home': 1}, 
                                 'total': {'total_under': 1}}}}]})
def test_get_lines(get, loads):
    assert model.get_lines()['team_name'][0] == 'hello'
