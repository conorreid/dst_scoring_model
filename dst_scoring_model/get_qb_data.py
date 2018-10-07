import logging

import requests
import pandas as pd
from bs4 import BeautifulSoup

import dst_scoring_model.maps as maps


def get_footballdb_data() -> pd.DataFrame:
    """Goes to the Football DB active QB website and grabs the data.

    First grabs interceptions percentages amongst all passing attempts, then
    grabs passing attempts per game to calculate interceptions per game. Then
    uses the qb_to_team mapping to map QB to a current team.
    """
    logging.info('grabbing interception percentage data')
    response = requests.get('https://www.pro-football-reference.com/leaders/pass_int_perc_active.htm')
    response_html = BeautifulSoup(response.text, 'html.parser')
    table = response_html.find('table', class_='sortable stats_table')
    interception_percentage_list = []
    for tr in table.find_all('tr')[1:]:
        tds = tr.find_all('td')
        interception_percentage_list.append(
            {'name': tds[0].text, 
             'interception_%': float(tds[1].text.strip('%'))/100})

    logging.info('grabbing passing attempts per game info')
    response = requests.get('https://www.pro-football-reference.com/leaders/pass_att_per_g_active.htm')
    response_html = BeautifulSoup(response.text, 'html.parser')
    table = response_html.find('table', class_='sortable stats_table')
    passing_attempts_list = []
    for tr in table.find_all('tr')[1:]:
        tds = tr.find_all('td')
        passing_attempts_list.append(
            {'name': tds[0].text, 
             'passing_attempts': float(tds[1].text)})
    interception_percentage_df = pd.DataFrame(data=interception_percentage_list)
    passing_attempts_df = pd.DataFrame(data=passing_attempts_list)

    logging.info('merging and mapping dataframes')
    merged_df = pd.merge(interception_percentage_df,
                         passing_attempts_df,
                         on='name', how='left')
    merged_df['interceptions_per_game_qb'] = (merged_df['interception_%'] *
                                              merged_df['passing_attempts'])
    merged_df['team_name'] = merged_df['name'].map(maps.qb_to_team)
    return merged_df