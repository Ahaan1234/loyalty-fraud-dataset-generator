"""DerivedFeatureEngine: per-account rolling-window ML features.

The inner loop is the performance-critical path — kept identical to the
original implementation; do not replace with pandas/numpy vectorised ops
without profiling first.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class DerivedFeatureEngine:
    """Computes rolling-window features and cumulative redemption ratios."""

    def compute(self, txn_df: pd.DataFrame, acc_df: pd.DataFrame) -> pd.DataFrame:
        print("  Computing derived features (this may take a moment for large datasets)...")

        txn_df = txn_df.sort_values(["account_id", "timestamp"]).reset_index(drop=True)
        txn_df = txn_df.merge(
            acc_df[["account_id", "account_age_days", "is_verified", "country"]],
            on="account_id",
            how="left",
        )

        features: list[pd.DataFrame] = []

        for acc_id, grp in txn_df.groupby("account_id"):
            grp = grp.sort_values("timestamp").copy()
            n   = len(grp)
            ts  = grp["timestamp"].values
            amt = grp["amount_usd"].values
            pts = grp["points_redeemed"].values

            txn_7d   = np.zeros(n, dtype=int)
            txn_30d  = np.zeros(n, dtype=int)
            pts_7d   = np.zeros(n, dtype=float)
            avg_30d  = np.zeros(n, dtype=float)
            days_lag = np.zeros(n, dtype=float)

            # ── Per-row rolling window (keep as-is — correct and fast enough) ─
            for idx in range(n):
                t   = ts[idx]
                w7  = (t - ts[:idx]) <= np.timedelta64(7,  "D")
                w30 = (t - ts[:idx]) <= np.timedelta64(30, "D")
                txn_7d[idx]   = w7.sum()
                txn_30d[idx]  = w30.sum()
                pts_7d[idx]   = pts[:idx][w7].sum()
                avg_30d[idx]  = amt[:idx][w30].mean() if w30.sum() > 0 else amt[idx]
                days_lag[idx] = (
                    (t - ts[idx - 1]) / np.timedelta64(1, "D") if idx > 0 else 999.0
                )

            features.append(pd.DataFrame({
                "txn_id":              grp["txn_id"].values,
                "txn_count_7d":        txn_7d,
                "txn_count_30d":       txn_30d,
                "pts_redeemed_7d":     pts_7d,
                "avg_spend_30d":       avg_30d.round(2),
                "days_since_last_txn": days_lag.round(2),
            }))

        feat_df = pd.concat(features, ignore_index=True)
        txn_df  = txn_df.merge(feat_df, on="txn_id", how="left")

        txn_df["net_pts_earned"]   = txn_df.groupby("account_id")["points_earned"].cumsum()
        txn_df["net_pts_redeemed"] = txn_df.groupby("account_id")["points_redeemed"].cumsum()
        txn_df["redemption_to_earn_ratio"] = (
            txn_df["net_pts_redeemed"] / txn_df["net_pts_earned"].clip(lower=1)
        ).round(4)
        txn_df = txn_df.drop(columns=["net_pts_earned", "net_pts_redeemed"])

        return txn_df
