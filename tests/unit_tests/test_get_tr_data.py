import unittest.mock as mock

import dst_scoring_model.get_tr_data as get_tr_data

HTML = """<html><body><table class="tr-table datatable scrollable">
<thead><tr><th>Rank</th></tr></thead><tbody>
<tr><td>1</td><td>Denver</td><td>4.0</td><td>3.3</td><td>4.0</td><td>4.0</td></tr>
<tr><td>2</td><td>Atlanta</td><td>--</td><td>3.0</td><td>4.0</td><td>2.8</td></tr>
</tbody></table></body></html>"""


@mock.patch("requests.get")
def test_get_tr_stats(get):
    get.return_value.text = HTML
    df = get_tr_data.get_tr_stats("http://x", "sacks")
    den = df[df.team_name == "Denver Broncos"].iloc[0]
    assert den["sacks_season"] == 4.0
    assert den["sacks_last_3"] == 3.3
    import numpy as np
    atl = df[df.team_name == "Atlanta Falcons"].iloc[0]
    assert np.isnan(atl["sacks_season"])


def test_blend_weight():
    assert get_tr_data.blend_weight(1) == 0.0
    assert get_tr_data.blend_weight(6) == 0.5
    assert get_tr_data.blend_weight(15) == 1.0
