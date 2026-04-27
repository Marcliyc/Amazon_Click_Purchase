from __future__ import annotations

import argparse
from pathlib import Path

from .data_prep import load_raw_data, make_daily_visits, make_session_time_visits, make_session_visits


def main(args=None):
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--visit-unit", default="daily", choices=["daily", "session"])
    ns = p.parse_args(args=args)

    out = Path(ns.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = load_raw_data(ns.input)
    sessions = make_session_visits(raw)
    visits = make_daily_visits(sessions) if ns.visit_unit == "daily" else make_session_time_visits(sessions)
    sessions.to_csv(out / "session_visits.csv", index=False)
    visits.to_csv(out / f"{ns.visit_unit}_visits.csv", index=False)


if __name__ == "__main__":
    main()
