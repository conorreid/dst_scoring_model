"""CLI wrapper: python dst_scoring_model.py [--week N] [--season YYYY] [--csv]"""
import argparse
import logging
from datetime import datetime

from dst_scoring_model import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--week", type=int, default=None)
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--csv", action="store_true",
                        help="also write df_final_<date>.csv")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    df = run(args.week, args.season)
    print(df.round(2).to_string(index=False))
    if args.csv:
        df.to_csv(f"df_final_{datetime.now():%Y_%m_%d}.csv", index=False)


if __name__ == "__main__":
    main()
