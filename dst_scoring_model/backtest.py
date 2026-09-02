"""Leak-free backtest and parameter fitting over past seasons.

Builds one row per team-week: stat rates as of that week (past data only),
the closing Vegas line, and the realized D/ST stats/fantasy score. Fitting
is staged: fusion weights per event minimize event-level MAE, then the
game-script coefficient is fit on the pass-pressure events, then the final
score metrics are reported.

Run:  python -m dst_scoring_model.backtest --seasons 2023 2024 2025 [--fit]
"""
import argparse
import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from dst_scoring_model import data, model, rates


def week_lines(games_df: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Implied points allowed + margin per team from closing lines.

    nflverse spread_line is expected home margin (positive = home favored):
      home implied = (total + spread) / 2, away implied = (total - spread) / 2
    """
    g = games_df[(games_df["season"] == season) & (games_df["week"] == week)]
    rows = []
    for gm in g.itertuples():
        total = gm.total_line if pd.notna(gm.total_line) else 44.0
        spread = gm.spread_line if pd.notna(gm.spread_line) else 0.0
        home_pts, away_pts = (total + spread) / 2, (total - spread) / 2
        rows.append({"team": gm.home_team, "opponent": gm.away_team,
                     "points_allowed": away_pts, "margin": home_pts - away_pts})
        rows.append({"team": gm.away_team, "opponent": gm.home_team,
                     "points_allowed": home_pts, "margin": away_pts - home_pts})
    return pd.DataFrame(data=rows)


def actuals(season: int) -> pd.DataFrame:
    """Realized per-team-week D/ST stats and standard-scoring fantasy points."""
    stats = data.team_week_stats(season)
    g = data.games()
    g = g[g["season"] == season]
    pa = {}
    for gm in g.itertuples():
        pa[(gm.week, gm.home_team)] = gm.away_score
        pa[(gm.week, gm.away_team)] = gm.home_score
    rows = []
    for r in stats.itertuples():
        allowed = pa.get((r.week, r.team))
        if allowed is None or pd.isna(allowed):
            continue
        rows.append({"week": r.week, "team": r.team, "act_pa": allowed,
                     "act_sacks": r.def_sacks, "act_ints": r.def_interceptions,
                     "act_fumbles": r.fumble_recovery_opp,
                     "act_tds": r.def_tds + r.special_teams_tds})
    df = pd.DataFrame(data=rows)
    scoring = model.Scoring()
    df["actual"] = df.apply(
        lambda r: scoring.score_actual(r["act_pa"], r["act_sacks"],
                                       r["act_ints"], r["act_fumbles"],
                                       r["act_tds"]), axis=1)
    return df


def build_dataset(seasons: list[int]) -> pd.DataFrame:
    """One row per team-week: as-of rates + line + realized outcome."""
    g = data.games()
    frames = []
    for season in seasons:
        act = actuals(season)
        for week in sorted(act["week"].unique()):
            rate_df = rates.rates_asof(season, int(week))
            lines = week_lines(g, season, int(week))
            lines = lines[lines["team"].isin(rate_df.index)
                          & lines["opponent"].isin(rate_df.index)]
            d = rate_df.loc[lines["team"]].reset_index(drop=True)
            o = rate_df.loc[lines["opponent"]].reset_index(drop=True)
            df = lines.reset_index(drop=True).copy()
            df["d_sacks"], df["o_sacks"] = d["def_sacks"].values, o["sacks_suffered"].values
            df["d_ints"], df["o_ints"] = (d["def_interceptions"].values,
                                          o["passing_interceptions"].values)
            df["d_ff"], df["o_fum"] = d["def_fumbles_forced"].values, o["off_fumbles"].values
            df["d_tds"] = (d["def_tds"] + d["special_teams_tds"]).values
            df["season"], df["week"] = season, week
            frames.append(df.merge(act[act["week"] == week], on=["week", "team"]))
    return pd.concat(frames, ignore_index=True)


def predictions(ds: pd.DataFrame, p: model.ModelParams) -> pd.DataFrame:
    """Vectorized event predictions (untruncated Poisson means)."""
    out = ds.copy()
    script = np.clip(1 + p.game_script_k * ds["margin"], 0.75, 1.25)
    out["sacks"] = (p.w_sacks * ds["d_sacks"]
                    + (1 - p.w_sacks) * ds["o_sacks"]) * script
    out["interceptions"] = (p.w_ints * ds["d_ints"]
                            + (1 - p.w_ints) * ds["o_ints"]) * script
    out["fumbles"] = (p.w_fumbles * ds["d_ff"]
                      + (1 - p.w_fumbles) * ds["o_fum"]) * p.fumble_recovery_rate
    out["defensive_touchdowns"] = ds["d_tds"]
    return out


def evaluate(ds: pd.DataFrame, p: model.ModelParams) -> dict:
    """Weekly rank correlation, score MAE, and top-5 streaming edge."""
    scoring = model.Scoring()
    pred = predictions(ds, p)
    pa_memo = {}

    def pa_score(x):
        key = round(x * 4) / 4
        if key not in pa_memo:
            pa_memo[key] = scoring.expected_pa_score(key)
        return pa_memo[key]

    pred["pred"] = (pred["points_allowed"].map(pa_score)
                    + pred["interceptions"] * scoring.interception
                    + pred["sacks"] * scoring.sack
                    + pred["fumbles"] * scoring.fumble
                    + pred["defensive_touchdowns"] * scoring.defensive_touchdown)
    corrs, top5_edge = [], []
    for (_, _), wk in pred.groupby(["season", "week"]):
        if len(wk) < 8:
            continue
        corrs.append(spearmanr(wk["pred"], wk["actual"]).statistic)
        top5 = wk.nlargest(5, "pred")["actual"].mean()
        top5_edge.append(top5 - wk["actual"].mean())
    return {"spearman": float(np.mean(corrs)),
            "mae": float((pred["pred"] - pred["actual"]).abs().mean()),
            "top5_edge": float(np.mean(top5_edge)),
            "weeks": len(corrs)}


def fit(ds: pd.DataFrame) -> model.ModelParams:
    """Stagewise grid fit: fusion weights on event MAE, then game script."""
    grid = np.round(np.arange(0, 1.01, 0.05), 2)

    def event_mae(pred_col, actual_col):
        return float((pred_col - ds[actual_col]).abs().mean())

    w_sacks = min(grid, key=lambda w: event_mae(
        w * ds["d_sacks"] + (1 - w) * ds["o_sacks"], "act_sacks"))
    w_ints = min(grid, key=lambda w: event_mae(
        w * ds["d_ints"] + (1 - w) * ds["o_ints"], "act_ints"))
    best_fum = min(
        ((w, r) for w in grid for r in np.arange(0.35, 0.66, 0.05)),
        key=lambda wr: event_mae(
            (wr[0] * ds["d_ff"] + (1 - wr[0]) * ds["o_fum"]) * wr[1],
            "act_fumbles"))
    # game script on the pass-pressure events with weights fixed
    def script_mae(k):
        s = np.clip(1 + k * ds["margin"], 0.75, 1.25)
        sacks = (w_sacks * ds["d_sacks"] + (1 - w_sacks) * ds["o_sacks"]) * s
        ints = (w_ints * ds["d_ints"] + (1 - w_ints) * ds["o_ints"]) * s
        return event_mae(sacks, "act_sacks") + event_mae(ints, "act_ints")

    k = min(np.round(np.arange(0, 0.061, 0.005), 3), key=script_mae)
    return model.ModelParams(w_sacks=float(w_sacks), w_ints=float(w_ints),
                             w_fumbles=float(best_fum[0]),
                             game_script_k=float(k),
                             fumble_recovery_rate=float(round(best_fum[1], 2)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+",
                        default=[2023, 2024, 2025])
    parser.add_argument("--fit", action="store_true",
                        help="fit parameters and write fitted_params.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    ds = build_dataset(args.seasons)
    print(f"dataset: {len(ds)} team-weeks over {args.seasons}")
    legacy = model.ModelParams(w_sacks=0.45, w_ints=0.45, w_fumbles=0.45,
                               game_script_k=0.0, fumble_recovery_rate=1.0)
    print("legacy-style params:", evaluate(ds, legacy))
    if args.fit:
        fitted = fit(ds)
        fitted.save()
        print("fitted params:", fitted)
        print("fitted metrics:", evaluate(ds, fitted))
    else:
        print("current params:", evaluate(ds, model.ModelParams.fitted()))


if __name__ == "__main__":
    main()
