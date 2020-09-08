import json
import logging

import pandas as pd
import requests


def get_lines() -> pd.DataFrame:
    """Calls out to rundown API to get Pinnacle betting lines.

    Returns:
        spread_df (pandas.Dataframe): df of the lines
    """
    logging.info("getting lines from Pinnacle")
    response = requests.get("https://therundown.io/api/v1/sports/2/events")
    events_list = json.loads(response.content.decode("utf-8"))
    spread_list = []

    for event in events_list["events"]:
        if event.get("schedule") and event["schedule"]["week"] == 1:
            spread_list.extend(
                [
                    {
                        "team_name": event["teams"][0]["name"],
                        "points_allowed": event["lines"]["3"]["total"]["total_under"]
                        / 2
                        + (-1) * event["lines"]["3"]["spread"]["point_spread_home"],
                        "opponent": event["teams"][1]["name"],
                    },
                    {
                        "team_name": event["teams"][1]["name"],
                        "points_allowed": event["lines"]["3"]["total"]["total_under"]
                        / 2
                        + event["lines"]["3"]["spread"]["point_spread_home"],
                        "opponent": event["teams"][0]["name"],
                    },
                ]
            )
    spread_df = pd.DataFrame(data=spread_list)
    spread_df["points_allowed"].fillna(27, inplace=True)
    return spread_df
