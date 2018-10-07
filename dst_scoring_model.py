import logging

import pandas as pd

import maps


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