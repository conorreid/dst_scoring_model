import json
import logging

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import requests
import scipy.stats as stats

import maps


def get_tr_stats(url: str, stat_name: str) -> pd.DataFrame:
    """Calls out to the URL specified at teamrankings.com, finds the table using
    BeautifulSoup, then sends table to construct_tr_df to make and return the
    dataframe.

    Args:
        stat_name (string): the stat name in question
        url (string): url of the site to scrape
    Returns:
        tr_df (pandas.Dataframe): dataframe of specified stat per team
    """
    logging.info('getting teamrankings stats for ' + stat_name)
    queried_stat = requests.get(url)
    queried_stat_html = BeautifulSoup(queried_stat.text, 'html.parser')
    table = queried_stat_html.find('table',
                                   class_='tr-table datatable scrollable')
    queried_stat_list = []
    for tr in table.find_all('tr')[1:]:
        tds = tr.find_all('td')
        try:
            queried_stat_list.append({'team_name': tds[1].text,
                                      stat_name + '_season': float(tds[2].text),
                                      stat_name + '_last_3': float(tds[3].text)})
        except ValueError:
            queried_stat_list.append({'team_name': tds[1].text,
                                      stat_name + '_season': np.nan,
                                      stat_name + '_last_3': np.nan})
    return construct_tr_df(queried_stat_list)


def get_tr_stats_full(url: str, stat_name: str) -> pd.DataFrame:
    """Calls out to URL for both 2017 and 2018 stats, fuses dfs, and returns
    a dataframe.

    Args:
        url (string): url of the site to scrape
        stat_name (string): the stat name in question
    Returns:
        tr_df (pandas.Dataframe): dataframe of specified stat per team
    """
    df1 = get_tr_stats(url, stat_name + '2018')
    df2 = get_tr_stats(url + '?date=2018-02-05', stat_name + '2017')
    df_merge = pd.merge(df1, df2, how='left',
                        on='team_name', suffixes=('', '_x'))
    df_merge[stat_name] = df_merge[stat_name + '2018_season'] * 0.5 + \
        df_merge[stat_name + '2018_last_3'] * 0.5
    df_merge = df_merge[['team_name', stat_name]]
    return df_merge


def construct_tr_df(tr_list: list) -> pd.DataFrame:
    """Takes in the list of dictionaries, converts to a dataframe, and then
    maps town_to_team dict as well as computes average of last season and last
    3 games score.

    Args:
        tr_list (list): list of dictionaries from tr_stats()
    Returns:
        tr_df (pandas.Dataframe): dataframe of specified stat per team
    """
    tr_df = pd.DataFrame(data=tr_list)
    tr_df['team_name'] = tr_df['team_name'].map(maps.town_to_team)
    return tr_df


def get_lines() -> pd.DataFrame:
    """Calls out to rundown API to get Pinnacle betting lines.

    Returns:
        spread_df (pandas.Dataframe): df of the lines
    """
    logging.info('getting lines from Pinnacle')
    response = requests.get('https://therundown.io/api/v1/sports/2/events')
    events_list = json.loads(response.content.decode('utf-8'))
    spread_list = []
    for event in events_list['events']:
        spread_list.extend([{'team_name': event['teams'][0]['name'],
                             'points_allowed': event['lines']['3']
                             ['total']['total_under']/2 + (-1) *
                             event['lines']['3']['spread']['point_spread_home'],
                             'opponent': event['teams'][1]['name']},
                            {'team_name': event['teams'][1]['name'],
                             'points_allowed': event['lines']['3']
                             ['total']['total_under']/2 +
                             event['lines']['3']['spread']['point_spread_home'],
                             'opponent': event['teams'][0]['name']}])
    spread_df = pd.DataFrame(data=spread_list)
    return spread_df


def defense_opponent_fusion(df: pd.DataFrame, stat: str) -> pd.DataFrame:
    """Create the composite number for each stat, fusing offense and defense.

    Args:
        df (pandas.Dataframe): df of the full table
        stat (string): stat to fuse
    Return:s
        df (pandas.Dataframe): df with new/dropped column(s)
    """
    logging.info('fusing stats for ' + stat)
    df[stat] = df[stat + '_created'] * 0.45 + df[stat + '_thrown'] * 0.55
    df = df.drop([stat + '_created', stat + '_thrown'], axis=1)
    return df


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


def main():
    spreads = get_lines()
    tr_items = {}
    defense_pure_df = pd.DataFrame()
    for key, value in maps.tr_stat_list.items():
        tr_items[key] = get_tr_stats_full(value['url'], value['column_name'])

    defense_pure_df = (pd.merge(spreads, tr_items['sacks_defense_list'],
                       how='left', on='team_name'))

    for item in maps.defense_pure_list:
        defense_pure_df = pd.merge(defense_pure_df, 
                                   tr_items[item],
                                   how='left', on='team_name')

    opponent_pure_df = pd.merge(tr_items['sacks_offense_list'],
                                tr_items['interceptions_thrown'],
                                how='left', on='team_name')
    opponent_pure_df = pd.merge(opponent_pure_df, tr_items['fumbles_thrown'],
                                how='left', on='team_name')

    fused_df = pd.merge(defense_pure_df, opponent_pure_df, how='left',
                        left_on='opponent', right_on='team_name')

    for key, value in maps.poisson_events.items():
        fused_df[key] = fused_df[key].apply(lambda row: poisson_create(row, 
                                                                       value))
    
    for item in maps.defense_opponent_list:
        fused_df = defense_opponent_fusion(fused_df, item)

    fused_df['points_allowed_score'] = fused_df['points_allowed'].apply(
        lambda row: points_allowed_score(row))
    logging.info('creating final score output')
    fused_df['final'] = fused_df['points_allowed_score'] + \
        fused_df['interceptions'] * 2 + fused_df['sacks'] * 1 + \
        fused_df['fumbles'] * 2 + fused_df['defensive_touchdowns'] * 6
    logging.info('formatting df')
    fused_df = fused_df.drop('team_name_y', axis=1)
    fused_df['team_name'] = fused_df['team_name_x']
    fused_df = fused_df[['team_name', 'final', 'opponent',
                         'points_allowed_score', 'interceptions',
                         'fumbles', 'sacks', 'defensive_touchdowns']]
    fused_df.sort_values(by='final', ascending=False, inplace=True)
    logging.info('creating csv')
    fused_df.to_csv('/home/conor/Documents/dst_scoring_model/df_final.csv',
                    index=False)


if __name__ == '__main__':
    main()
