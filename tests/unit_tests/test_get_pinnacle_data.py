import unittest.mock as mock

import dst_scoring_model.get_pinnacle_data as get_pinnacle_data


def construct_request_content():
    response = mock.MagicMock()
    response.content = mock.MagicMock(return_value='hello')
    return response


@mock.patch('requests.get', return_value=construct_request_content())
@mock.patch('json.loads',
            return_value={'events': [
                {'teams': [
                    {'name': 'hello'}, {'name': 'goodbye'}],
                 'lines': {'3': {'spread': {'point_spread_home': 1},
                                 'total': {'total_under': 1}}}}]})
def test_get_lines(get, loads):
    assert get_pinnacle_data.get_lines()['team_name'][0] == 'hello'
