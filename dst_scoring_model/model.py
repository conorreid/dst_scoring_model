import numpy as np
import scipy.stats as stats


def poisson_create(rate: float, max_possible: float) -> float:
    """Creates poisson distribution by making possibility matrix of values and
    then adding together matrix to find predicted value of event.

    Args:
        rate (float): the rate at which the event in question occurs on average
        max_possible (float): the max possible number of times event can occur
    Returns:
        event_pred (float): the sum of all the predicted events and their rates
    """
    n = np.arange(0, max_possible)
    n2 = np.arange(0, max_possible)
    y = stats.poisson.pmf(n, rate)
    y2 = n2 * y
    event_pred = y2.sum()
    return event_pred


def points_allowed_score(points: float) -> int:
    """The points allowed score logic generator based on standard D/ST fantasy
    football scoring.

    Args:
        points (float): number of points allowed
    Returns:
        score (int): the score got for that number of points allowed
    """
    if points == 0:
        score = 10
    elif points < 7:
        score = 7
    elif points < 14:
        score = 4
    elif points < 18:
        score = 1
    elif points < 28:
        score = 0
    elif points < 35:
        score = -3
    else:
        score = -4
    return score
