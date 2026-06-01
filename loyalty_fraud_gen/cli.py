"""CLI entrypoint — orchestrates account generation, transaction generation,
dirtiness, feature engineering, and output."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from faker import Faker

from loyalty_fraud_gen.config import Config
from loyalty_fraud_gen.generators import AccountGenerator, OnlineFraudGenerator, InStoreFraudGenerator
from loyalty_fraud_gen.pipeline import DerivedFeatureEngine, DirtinessLayer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic loyalty program fraud dataset (v3)"
    )
    parser.add_argument("--records",       type=int,   default=50_000)
    parser.add_argument("--fraud-rate",    type=float, default=0.05)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--out",           type=str,   default="loyalty_fraud_data.csv")
    parser.add_argument("--mode",          type=str,   default="clean",
                        choices=["clean", "dirty"],
                        help="clean = perfect labels; dirty = FP/FN edge cases + data noise")
    parser.add_argument("--dirt-level",    type=str,   default="medium",
                        choices=["low", "medium", "high"],
                        help="Intensity of dirtiness (only applies in dirty mode)")
    parser.add_argument("--no-features",   action="store_true",
                        help="Skip derived feature engineering")
    parser.add_argument("--channel-split", type=float, default=0.35,
                        help="Fraction of records that are in-store (default: 0.35)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    Faker.seed(args.seed)
    fake   = Faker()
    config = Config()

    _print_header(args)

    # ── [1/5] Accounts ────────────────────────────────────────────────────────
    n_legit_accts = max(200, int(args.records * 0.015))
    n_rings       = max(5,   int(args.records * args.fraud_rate * 0.004))
    print(f"[1/5] Generating accounts ({n_legit_accts:,} legit + {n_rings} referral rings)...")
    acct_gen = AccountGenerator(rng, config, fake)
    accounts_df, _, store_pools = acct_gen.generate(n_legit_accts, n_rings)
    print(f"      Total accounts: {len(accounts_df):,}  |  stores: {len(store_pools)}")

    # ── [2/5] Transactions ────────────────────────────────────────────────────
    channel_split = max(0.0, min(1.0, args.channel_split))
    n_instore     = int(args.records * channel_split)
    n_online      = args.records - n_instore
    print(f"\n[2/5] Generating transactions  "
          f"(online: {n_online:,}  in-store: {n_instore:,})...")

    online_gen  = OnlineFraudGenerator(rng, config, fake)
    online_rows, next_id = online_gen.generate(
        accounts_df, n_online, args.fraud_rate, args.mode, txn_id_start=100_000
    )

    instore_gen  = InStoreFraudGenerator(rng, config, fake)
    instore_rows, _ = instore_gen.generate(
        accounts_df, n_instore, args.fraud_rate, args.mode, store_pools,
        txn_id_start=next_id,
        existing_txns=online_rows or None,
    )

    all_rows = online_rows + instore_rows
    txn_df   = pd.DataFrame(all_rows)
    txn_df["timestamp"] = pd.to_datetime(txn_df["timestamp"])
    txn_df = txn_df.sort_values("timestamp").reset_index(drop=True)
    print(f"      Rows generated: {len(txn_df):,}")

    _print_fraud_breakdown(txn_df, args.mode)

    # ── [3/5] Dirtiness ───────────────────────────────────────────────────────
    if args.mode == "dirty":
        print(f"\n[3/5] Applying dirtiness (level: {args.dirt_level})...")
        dirt = DirtinessLayer(rng, config)
        txn_df = dirt.apply(txn_df, args.dirt_level)
        dirt_counts = (
            txn_df[txn_df["dirt_applied"] != "none"]["dirt_applied"]
            .str.split("|").explode().value_counts()
        )
        for dtype, cnt in dirt_counts.items():
            print(f"        {dtype:<25}: {cnt:>6,} rows affected")
        print(f"      Total rows after duplicates: {len(txn_df):,}")
    else:
        print(f"\n[3/5] Mode = clean, skipping dirtiness layer.")

    # ── [4/5] Features ────────────────────────────────────────────────────────
    if not args.no_features:
        print(f"\n[4/5] Computing derived ML features...")
        engine = DerivedFeatureEngine()
        txn_df = engine.compute(txn_df, accounts_df)
    else:
        print(f"\n[4/5] Skipping derived features (--no-features).")

    # ── [5/5] Save ────────────────────────────────────────────────────────────
    print(f"\n[5/5] Saving outputs...")
    txn_df.to_csv(args.out, index=False)
    accts_out = "accounts_" + args.out
    accounts_df.to_csv(accts_out, index=False)

    print(f"\n  ✓  Transactions : {args.out}  ({len(txn_df):,} rows × {len(txn_df.columns)} cols)")
    print(f"  ✓  Accounts     : {accts_out}  ({len(accounts_df):,} rows)")
    print(f"\n  All columns in transactions file:")
    for col in txn_df.columns:
        print(f"    - {col}")
    print(f"\n{'='*58}")
    print(f"  Done.")
    print(f"{'='*58}\n")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_header(args: argparse.Namespace) -> None:
    print(f"\n{'='*58}")
    print(f"  Loyalty Program Fraud Data Generator  v3.0")
    print(f"{'='*58}")
    print(f"  Records      : {args.records:,}")
    print(f"  Fraud rate   : {args.fraud_rate:.1%}")
    print(f"  Mode         : {args.mode}"
          + (f"  [{args.dirt_level}]" if args.mode == "dirty" else ""))
    print(f"  Channel split: {args.channel_split:.0%} in-store")
    print(f"  Seed         : {args.seed}")
    print(f"  Output       : {args.out}")
    print(f"{'='*58}\n")


def _print_fraud_breakdown(txn_df: pd.DataFrame, mode: str) -> None:
    fraud_df = txn_df[txn_df["is_fraud"]]
    print(f"\n      Fraud breakdown by type × channel (pre-dirt):")

    if len(fraud_df) > 0:
        breakdown = (
            fraud_df.groupby(["fraud_type", "channel"])
            .size()
            .reset_index(name="count")
        )
        for _, row in breakdown.iterrows():
            pct = row["count"] / len(txn_df)
            print(f"        {row['fraud_type']:<32} [{row['channel']:<8}]: "
                  f"{row['count']:>5,}  ({pct:.2%})")
    total_fraud = fraud_df.shape[0]
    print(f"        {'Total fraud':<32}           : "
          f"{total_fraud:>5,}  ({total_fraud/len(txn_df):.2%})")

    if mode == "dirty":
        print(f"\n      Edge cases (FP/FN):")
        fp_counts = txn_df[txn_df["fp_risk_flag"] != "none"]["fp_risk_flag"].value_counts()
        fn_counts = txn_df[txn_df["fn_risk_flag"] != "none"]["fn_risk_flag"].value_counts()
        for flag, cnt in fp_counts.items():
            print(f"        FP  {flag:<30}: {cnt:>5,}")
        for flag, cnt in fn_counts.items():
            print(f"        FN  {flag:<30}: {cnt:>5,}")


if __name__ == "__main__":
    main()
