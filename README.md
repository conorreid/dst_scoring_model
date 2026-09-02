# D/ST scoring model

Fantasy football D/ST streaming model, loosely based on the framework from
quickonthedrawl on r/fantasyfootball
(https://www.reddit.com/r/fantasyfootball/comments/9cv5of/building_and_improving_on_existing_dst_projections/).

## Methodology
For each defense with a game this week:
- **Points allowed** comes from Vegas lines (ESPN scoreboard API / DraftKings):
  the opponent's implied total from the spread and over/under. The fantasy
  points-allowed score is the *expectation of the bracket score over a Poisson
  distribution* around that implied total, not the bracket of the mean.
- **Sacks, interceptions, fumbles, defensive TDs** are Poisson-expected counts
  from per-game rates (teamrankings.com), fusing what the defense creates
  (45%) with what the opponent gives up (55%).
- Rates blend the **previous season** (via dated teamrankings tables) with the
  **current season** (half season-to-date, half last-3-games), ramping from
  all-prior in week 1 to all-current by week 11.
- Scoring is configurable per league (`model.Scoring`,
  `model.scoring_from_sleeper`).

## Usage
```
pip install -e .
python dst_scoring_model.py --week 1 --season 2026        # print rankings
python dst_scoring_model.py --csv                          # current week + csv
```
```python
from dst_scoring_model import run, Scoring
df = run(week=1, season=2026)   # expected events + standard 'final' score
```

## Sources
- ESPN scoreboard API for betting lines (replaces the defunct
  therundown.io/Pinnacle source)
- teamrankings.com for team stat rates
