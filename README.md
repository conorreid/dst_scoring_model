# D/ST scoring model

Fantasy football D/ST streaming model: Vegas-implied points allowed plus
Poisson-expected defensive events, with parameters fitted by backtest.

## Methodology
For each defense with a game this week:
- **Points allowed**: the opponent's implied total from the spread and
  over/under (ESPN scoreboard API live; nflverse closing lines in backtests).
  The fantasy PA score is the expectation of the bracket score over a Poisson
  distribution around the implied total.
- **Sacks & interceptions**: fusion of the defense's own rate and the
  opponent's giveaway rate (weights fitted by backtest — the opponent's
  sack-proneness matters more than the defense's rate), scaled by a fitted
  game-script factor (favored defenses face more dropbacks). A home-field
  pressure boost was tested and fit to exactly 1.0 — no effect.
- **Fumbles**: forced-fumble rate times a fitted league recovery rate —
  recovering fumbles is luck; forcing them is skill.
- **Touchdowns**: a fitted conversion rate per expected turnover (raw
  defensive-TD averages are too noisy to use), plus the special-teams
  return TD rate. A compound 7s-and-3s points distribution was also tested
  against plain Poisson and did not improve bracket scoring.
- Rates come from nflverse weekly team stats, blending last season with the
  current one (half season-to-date, half last-3), ramping over 10 weeks.
- Scoring configurable per league (`model.Scoring`, `scoring_from_sleeper`).

## Backtest
`python -m dst_scoring_model.backtest --seasons 2023 2024 2025 [--fit]`
builds a leak-free dataset (rates as of each week, closing lines, realized
stats) and reports weekly Spearman, MAE, and the top-5 streaming edge.
`--fit` grid-fits the fusion weights, game-script coefficient, and fumble
recovery rate, writing `fitted_params.json` (shipped with the package).

2023-25 (1,632 team-weeks), legacy params -> fitted: Spearman .272 -> .310,
MAE 4.37 -> 4.23, top-5 streaming edge +2.09 -> +2.24 pts/week.

## Usage
```
pip install -e .
python dst_scoring_model.py --week 1 --season 2026
```
```python
from dst_scoring_model import run, Scoring
df = run(week=1, season=2026)
```

## Sources
- ESPN scoreboard API (DraftKings) for live betting lines
- nflverse (stats_team_week, games.csv) for stat rates, historical lines,
  and backtest actuals
