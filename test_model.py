import model
import pytest
import pandas as pd
import unittest.mock as mock


def construct_request_content():
    response = mock.MagicMock()
    response.content = mock.MagicMock(return_value='hello')
    return response


def construct_request_content_tr():
    response = mock.MagicMock()
    response.text = """<html>
    <body>
        <main>
            <table class="tr-table datatable scrollable">
                <thead>
                    <tr>
                        <th class="text-center sort-asc-first sort-first">Rank</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td class="rank text-center" data-sort="1">1</td>
                        <td class="text-left nowrap" data-sort="LA Rams"><a href="https://www.teamrankings.com/nfl/team/los-angeles-rams">LA
                                Rams</a></td>
                        <td class="text-right" data-sort="0.411765">0.4</td>
                        <td class="text-right" data-sort="0">0.0</td>
                        <td class="text-right" data-sort="0">0.0</td>
                        <td class="text-right" data-sort="0.5">0.5</td>
                        <td class="text-right" data-sort="0.333333">0.3</td>
                        <td class="text-right" data-sort="1.25">1.2</td>
                    </tr>
                    <tr>
                        <td class="rank text-center" data-sort="1">1</td>
                        <td class="text-left nowrap" data-sort="LA Rams"><a href="https://www.teamrankings.com/nfl/team/los-angeles-rams">LA
                                Rams</a></td>
                        <td class="text-right" data-sort="0.411765">0.4</td>
                        <td class="text-right" data-sort="0">0.0</td>
                        <td class="text-right" data-sort="0">0.0</td>
                        <td class="text-right" data-sort="0.5">0.5</td>
                        <td class="text-right" data-sort="0.333333">0.3</td>
                        <td class="text-right" data-sort="1.25">1.2</td>
                    </tr>
                </tbody>
        </main>>
    </body>
    </html>"""
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


def test_construct_tr_df():
    tr_list = [{'team_name': 'Arizona'}]
    assert model.construct_tr_df(tr_list)['team_name'][0] == 'Arizona Cardinals'


@mock.patch('model.get_tr_stats',
            return_value=pd.DataFrame(data={'team_name': ['hello'],
                                            'stat2018_season': [1],
                                            'stat2018_last_3': [1]}))
def test_get_tr_stats_full(get_tr_stats):
    assert model.get_tr_stats_full('hello', 'stat')['team_name'][0] == 'hello'


@mock.patch('requests.get', return_value=construct_request_content_tr())
def test_get_tr_stats(get):
    assert model.get_tr_stats('test', 'hello')['hello_season'][0] == 0.4

