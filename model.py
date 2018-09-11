import requests
import json
from bs4 import BeautifulSoup
import numpy as np
import scipy.stats as stats
import pandas as pd
import maps

def get_tr_stats(url, stat_name):
	sacks_defense = requests.get(url)
	sacks_defense_html = BeautifulSoup(sacks_defense.text, 'html.parser')
	table = sacks_defense_html.find('table', class_ = 'tr-table datatable scrollable')
	sacks_defense_list = []
	for tr in table.find_all('tr')[1:]:
		tds = tr.find_all('td')
		sacks_defense_list.append({'team_name': tds[1].text,
								   stat_name + '_season': float(tds[2].text),
								   stat_name + '_last_3': float(tds[3].text)})
	sacks_defense_list = pd.DataFrame(data=sacks_defense_list)
	sacks_defense_list['team_name'] = sacks_defense_list['team_name'].map(maps.town_to_team)
	sacks_defense_list[stat_name] = sacks_defense_list[stat_name + '_season'] * 0.5 + sacks_defense_list[stat_name + '_last_3'] * 0.5
	sacks_defense_list = sacks_defense_list.drop([stat_name + '_season', stat_name + '_last_3'], axis=1)
	return sacks_defense_list


def get_lines():
	response = requests.get('https://therundown.io/api/v1/sports/2/events')
	events_list = json.loads(response.content)
	spread_list = []
	for event in events_list['events']:
		spread_list.extend([{'team_name': event['teams'][0]['name'], 
						   'points_allowed': (
						   	   event['lines']['3']['total']['total_under']/2) + 
							   (-1) * event['lines']['3']['spread']['point_spread_home'],
							'opponent': event['teams'][1]['name']},
						  {'team_name': event['teams'][1]['name'], 
						   'points_allowed': (
						       event['lines']['3']['total']['total_under']/2) + 
						       event['lines']['3']['spread']['point_spread_home'],
						    'opponent': event['teams'][0]['name']}])
	spread_df = pd.DataFrame(data=spread_list)
	return spread_df


def defense_opponent_fusion(df, stat):
	df[stat] = df[stat + '_created'] * 0.45 + df[stat + '_thrown'] * 0.55
	df = df.drop([stat + '_created', stat + '_thrown'], axis=1)
	return df


def poisson_create(rate, max_possible):
	n = np.arange(0, max_possible)
	n2 = np.arange(0, max_possible)
	y = stats.poisson.pmf(n, rate)
	y2 = n2 * y
	return y2.sum()

def points_allowed_score(points):
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


spreads = get_lines()
sacks_defense_list = get_tr_stats('https://www.teamrankings.com/nfl/stat/sacks-per-game', 'sacks_created')
sacks_offense_list = get_tr_stats('https://www.teamrankings.com/nfl/stat/qb-sacked-per-game', 'sacks_thrown')
interceptions_thrown = get_tr_stats('https://www.teamrankings.com/nfl/stat/interceptions-thrown-per-game', 
									'interceptions_thrown')
interceptions_created = get_tr_stats('https://www.teamrankings.com/nfl/stat/interceptions-per-game',
									 'interceptions_created')
fumbles_thrown = get_tr_stats('https://www.teamrankings.com/nfl/stat/fumbles-per-game', 'fumbles_thrown')
fumbles_created = get_tr_stats('https://www.teamrankings.com/nfl/stat/opponent-fumbles-per-game', 'fumbles_created')
defensive_touchdowns = get_tr_stats('https://www.teamrankings.com/nfl/stat/defensive-touchdowns-per-game',
									'defensive_touchdowns')

defense_pure_df = pd.merge(spreads, sacks_defense_list, how='left', on='team_name')
defense_pure_df = pd.merge(defense_pure_df, interceptions_created, how='left', on='team_name')
defense_pure_df = pd.merge(defense_pure_df, fumbles_created, how='left', on='team_name')
defense_pure_df = pd.merge(defense_pure_df, defensive_touchdowns, how='left', on='team_name')
opponent_pure_df = pd.merge(sacks_offense_list, interceptions_thrown, how='left', on='team_name')
opponent_pure_df = pd.merge(opponent_pure_df, fumbles_thrown, how='left', on='team_name')
fused_df = pd.merge(defense_pure_df, opponent_pure_df, how='left', left_on='opponent', right_on='team_name')

fused_df['points_allowed'] = fused_df['points_allowed'].apply(lambda row: poisson_create(row, 50))
fused_df['sacks_created'] = fused_df['sacks_created'].apply(lambda row: poisson_create(row, 7))
fused_df['interceptions_created'] = fused_df['interceptions_created'].apply(lambda row: poisson_create(row, 7))
fused_df['fumbles_created'] = fused_df['fumbles_created'].apply(lambda row: poisson_create(row, 5))
fused_df['defensive_touchdowns'] = fused_df['defensive_touchdowns'].apply(lambda row: poisson_create(row, 4))
fused_df['sacks_thrown'] = fused_df['sacks_created'].apply(lambda row: poisson_create(row, 7))
fused_df['interceptions_thrown'] = fused_df['interceptions_created'].apply(lambda row: poisson_create(row, 7))
fused_df['fumbles_thrown'] = fused_df['fumbles_created'].apply(lambda row: poisson_create(row, 5))
fused_df = defense_opponent_fusion(fused_df, 'fumbles')
fused_df = defense_opponent_fusion(fused_df, 'interceptions')
fused_df = defense_opponent_fusion(fused_df, 'sacks')
fused_df['points_allowed_score'] = fused_df['points_allowed'].apply(lambda row: points_allowed_score(row))
fused_df['final'] = fused_df['points_allowed_score'] + fused_df['interceptions'] * 2 + fused_df['sacks'] * 1 + fused_df['fumbles'] * 2 + fused_df['defensive_touchdowns'] * 6
fused_df = fused_df.drop('team_name_y', axis=1)
fused_df.to_csv('df_final.csv')