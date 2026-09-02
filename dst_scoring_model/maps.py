town_to_team = {
    "Arizona": "Arizona Cardinals",
    "Atlanta": "Atlanta Falcons",
    "Baltimore": "Baltimore Ravens",
    "Buffalo": "Buffalo Bills",
    "Carolina": "Carolina Panthers",
    "Chicago": "Chicago Bears",
    "Cincinnati": "Cincinnati Bengals",
    "Cleveland": "Cleveland Browns",
    "Dallas": "Dallas Cowboys",
    "Denver": "Denver Broncos",
    "Detroit": "Detroit Lions",
    "Green Bay": "Green Bay Packers",
    "Houston": "Houston Texans",
    "Indianapolis": "Indianapolis Colts",
    "Jacksonville": "Jacksonville Jaguars",
    "Kansas City": "Kansas City Chiefs",
    "LA Chargers": "Los Angeles Chargers",
    "LA Rams": "Los Angeles Rams",
    "Miami": "Miami Dolphins",
    "Minnesota": "Minnesota Vikings",
    "New England": "New England Patriots",
    "New Orleans": "New Orleans Saints",
    "NY Giants": "New York Giants",
    "NY Jets": "New York Jets",
    "Las Vegas": "Las Vegas Raiders",
    "Philadelphia": "Philadelphia Eagles",
    "Pittsburgh": "Pittsburgh Steelers",
    "San Francisco": "San Francisco 49ers",
    "Seattle": "Seattle Seahawks",
    "Tampa Bay": "Tampa Bay Buccaneers",
    "Tennessee": "Tennessee Titans",
    "Washington": "Washington Commanders",
}

# full team name -> common abbreviation, for consumers that key by code
team_to_abbrev = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Las Vegas Raiders": "LV",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WSH",
}

tr_stat_list = {
    "sacks_defense_list": {
        "url": "https://www.teamrankings.com/nfl/stat/sacks-per-game",
        "column_name": "sacks_created",
    },
    "sacks_offense_list": {
        "url": "https://www.teamrankings.com/nfl/stat/qb-sacked-per-game",
        "column_name": "sacks_thrown",
    },
    "interceptions_thrown": {
        "url": "https://www.teamrankings.com/nfl/stat/interceptions-thrown-per-game",
        "column_name": "interceptions_thrown",
    },
    "interceptions_created": {
        "url": "https://www.teamrankings.com/nfl/stat/interceptions-per-game",
        "column_name": "interceptions_created",
    },
    "fumbles_thrown": {
        "url": "https://www.teamrankings.com/nfl/stat/fumbles-per-game",
        "column_name": "fumbles_thrown",
    },
    "fumbles_created": {
        "url": "https://www.teamrankings.com/nfl/stat/opponent-fumbles-per-game",
        "column_name": "fumbles_created",
    },
    "defensive_touchdowns": {
        "url": "https://www.teamrankings.com/nfl/stat/defensive-touchdowns-per-game",
        "column_name": "defensive_touchdowns",
    },
}

defense_pure_list = ["interceptions_created", "fumbles_created", "defensive_touchdowns"]

poisson_events = {
    "sacks_created": 7,
    "interceptions_created": 7,
    "fumbles_created": 5,
    "defensive_touchdowns": 4,
    "sacks_thrown": 7,
    "interceptions_thrown": 7,
    "fumbles_thrown": 5,
}

defense_opponent_list = ["fumbles", "interceptions", "sacks"]
