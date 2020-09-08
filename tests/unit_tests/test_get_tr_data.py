import unittest.mock as mock

import pandas as pd

import dst_scoring_model.get_tr_data as get_tr_data


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


def test_construct_tr_df():
    tr_list = [{"team_name": "Arizona"}]
    assert get_tr_data.construct_tr_df(tr_list)["team_name"][0] == "Arizona Cardinals"


@mock.patch(
    "dst_scoring_model.get_tr_data.get_tr_stats",
    return_value=pd.DataFrame(
        data={"team_name": ["hello"], "stat2019_season": [1], "stat2019_last_3": [1]}
    ),
)
def test_get_tr_stats_full(get_tr_stats):
    assert get_tr_data.get_tr_stats_full("hello", "stat")["team_name"][0] == "hello"


@mock.patch("requests.get", return_value=construct_request_content_tr())
def test_get_tr_stats(get):
    assert get_tr_data.get_tr_stats("test", "hello")["hello_season"][0] == 0.4
