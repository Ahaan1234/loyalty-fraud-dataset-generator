"""DirtinessLayer: null injection, duplicates, timestamp jitter, label noise, points drift."""
from __future__ import annotations

import numpy as np
import pandas as pd

from loyalty_fraud_gen.config import Config


class DirtinessLayer:
    """Apply realistic data-quality issues on top of a clean transaction table."""

    def __init__(self, rng: np.random.Generator, config: Config) -> None:
        self.rng = rng
        self.cfg = config

    def apply(self, df: pd.DataFrame, dirt_level: str) -> pd.DataFrame:
        """Return a new DataFrame with dirtiness applied.

        Adds a `dirt_applied` audit column that records which operations
        touched each row (pipe-delimited).
        """
        null_rate, dup_rate, jitter_sec, label_noise_rate, pts_drift_rate = (
            self.cfg.dirt_params[dirt_level]
        )
        df = df.copy()
        df["dirt_applied"] = "none"
        n = len(df)

        # 1. Null device / IP fields
        null_mask_device = self.rng.random(n) < null_rate
        null_mask_ip     = self.rng.random(n) < (null_rate * 0.5)
        df.loc[null_mask_device, "device_fingerprint"] = None
        df.loc[null_mask_ip,     "login_ip"]           = None
        changed = null_mask_device | null_mask_ip
        df.loc[changed, "dirt_applied"] = df.loc[changed, "dirt_applied"].apply(
            lambda x: "null_fields" if x == "none" else x + "|null_fields"
        )

        # 2. Timestamp jitter
        jitter_mask = self.rng.random(n) < 0.15
        jitter_secs = self.rng.integers(-jitter_sec, jitter_sec, size=n)
        ts_array    = pd.to_datetime(df["timestamp"])
        jitter_td   = pd.to_timedelta(jitter_secs * jitter_mask.astype(int), unit="s")
        df["timestamp"] = ts_array + jitter_td
        df.loc[jitter_mask, "dirt_applied"] = df.loc[jitter_mask, "dirt_applied"].apply(
            lambda x: "ts_jitter" if x == "none" else x + "|ts_jitter"
        )

        # 3. Duplicate transactions (network retries)
        idx_all  = df.index.tolist()
        n_dups   = max(1, int(n * dup_rate))
        dup_idx  = self.rng.choice(idx_all, size=n_dups, replace=False)
        dup_rows = df.loc[dup_idx].copy()
        max_txn  = df["txn_id"].max()
        dup_rows["txn_id"]    = range(max_txn + 1, max_txn + 1 + n_dups)
        dup_rows["timestamp"] = dup_rows["timestamp"] + pd.to_timedelta(
            self.rng.integers(1, 30, size=n_dups), unit="s"
        )
        dup_rows["dirt_applied"] = "duplicate"
        df = pd.concat([df, dup_rows], ignore_index=True)

        # 4. Points arithmetic drift
        drift_mask = self.rng.random(len(df)) < pts_drift_rate
        drift_pct  = self.rng.uniform(0.85, 1.30, size=len(df))
        df.loc[drift_mask, "points_earned"] = (
            df.loc[drift_mask, "points_earned"] * drift_pct[drift_mask]
        ).astype(int)
        df.loc[drift_mask, "dirt_applied"] = df.loc[drift_mask, "dirt_applied"].apply(
            lambda x: "pts_drift" if x == "none" else x + "|pts_drift"
        )

        # 5. Label noise — flip recent fraud → legit ("unconfirmed" lag)
        recent_cutoff = df["timestamp"].max() - pd.Timedelta(days=30)
        recent_fraud  = df[(df["is_fraud"]) & (df["timestamp"] >= recent_cutoff)].index
        if len(recent_fraud) > 0:
            n_flip  = max(1, int(len(recent_fraud) * label_noise_rate))
            flip_idx = self.rng.choice(recent_fraud, size=min(n_flip, len(recent_fraud)), replace=False)
            df.loc[flip_idx, "is_fraud"]   = False
            df.loc[flip_idx, "fraud_type"] = "none"
            df.loc[flip_idx, "dirt_applied"] = df.loc[flip_idx, "dirt_applied"].apply(
                lambda x: "label_noise" if x == "none" else x + "|label_noise"
            )

        df = df.sort_values("timestamp").reset_index(drop=True)
        return df
