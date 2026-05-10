from __future__ import annotations

from typing import Any

import pandas as pd


def identify_early_first_purchase_customers(
    sessions: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Mark left-censored customers whose first observed purchase is in week 1..N.

    These customers are not treated as a normal acquisition cohort for the
    Transformer because their first observed purchase occurs during the initial
    observation window and may not be their true first purchase.  They are sent
    to the no-covariate EV/CM segment instead.
    """
    customer_col = config["data"]["customer_id_col"]
    early_weeks = int(config.get("evcm", {}).get("early_purchase_weeks", 4))
    min_week = pd.to_datetime(sessions["calendar_week"]).min()
    cutoff_week = min_week + pd.Timedelta(weeks=max(early_weeks - 1, 0))
    first_purchase = (
        sessions.loc[sessions["transaction"] > 0]
        .groupby(customer_col, as_index=False)["calendar_week"]
        .min()
        .rename(columns={"calendar_week": "first_purchase_week"})
    )
    customers = sessions[[customer_col]].drop_duplicates().merge(first_purchase, on=customer_col, how="left")
    customers["cbmt_segment"] = "transformer"
    early_mask = customers["first_purchase_week"].notna() & (pd.to_datetime(customers["first_purchase_week"]) <= cutoff_week)
    customers.loc[early_mask, "cbmt_segment"] = "early_purchase_evcm"
    customers["early_purchase_cutoff_week"] = cutoff_week
    return customers
