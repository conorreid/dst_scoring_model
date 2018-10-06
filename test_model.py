import model
import pytest

@pytest.mark.parametrize("test_input, expect", [(0, 10), (6, 7), (13, 4),
                                                (17, 1), (27, 0), (34, -3),
                                                (36, -4)])
def test_points_allowed_score(test_input: int, expect: int) -> bool:
    assert model.points_allowed_score(test_input) == expect
