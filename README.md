# Loyalty Program Fraud Dataset Generator

Generates synthetic loyalty-program transaction data with labelled fraud,
deliberate false-positive / false-negative edge cases, optional data dirtiness,
and both **online** and **in-store** channels.

## Package structure

```
loyalty_fraud_gen/
├── __init__.py
├── config.py          # All constants + Config dataclass
├── models.py          # Transaction dataclass
├── generators/
│   ├── base.py        # BaseGenerator (weighted_choice, random_ip, …)
│   ├── accounts.py    # AccountGenerator — accounts DataFrame + store pools
│   ├── online.py      # OnlineFraudGenerator — ATO, referral, redemption abuse, FP/FN
│   └── instore.py     # InStoreFraudGenerator — 4 in-store fraud types + FP/FN
├── pipeline/
│   ├── features.py    # DerivedFeatureEngine — rolling-window ML features
│   └── dirtiness.py   # DirtinessLayer — nulls, duplicates, jitter, label noise
└── cli.py             # Thin argparse entrypoint + summary report
```

A root-level `generate_loyalty_fraud_data.py` shim calls `loyalty_fraud_gen.cli.main()`
for backwards compatibility.

## CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--records N` | 50 000 | Total rows to generate (before dirtiness duplicates) |
| `--fraud-rate F` | 0.05 | Fraction of records that are fraud |
| `--seed S` | 42 | Random seed for full reproducibility |
| `--out FILE` | `loyalty_fraud_data.csv` | Output CSV path (accounts written to `accounts_<FILE>`) |
| `--mode MODE` | `clean` | `clean` = perfect labels; `dirty` = FP/FN + data noise |
| `--dirt-level LEVEL` | `medium` | `low` / `medium` / `high` — only used in dirty mode |
| `--no-features` | off | Skip the rolling-window derived feature computation |
| `--channel-split F` | 0.35 | Fraction of records that are **in-store** (remainder = online) |

## Fraud types

### Online channel
| Type | Description |
|------|-------------|
| `account_takeover` | Credential-stuffed account; foreign IP, bot device, high redemption |
| `referral_fraud` | Fake accounts created to farm referral bonuses |
| `redemption_abuse` | Buy high-value item with points, then return for cash |

### In-store channel
| Type | Description |
|------|-------------|
| `card_id_swap` | Fraudster uses another person's physical loyalty card at an unfamiliar store during off-peak hours |
| `employee_collusion` | Cashier posts phantom zero-value transactions to award accomplices with high points |
| `receipt_manipulation` | Fraudster submits a return using a forged/stolen receipt from a different account |
| `terminal_hopping` | Same account hits 4–8 distinct stores within 24–72 hours (geographic velocity) |

### In-store-only columns
Populated for `in_store` rows, `None` for online:

| Column | Example | Notes |
|--------|---------|-------|
| `pos_terminal_id` | `TERM-042` | Fixed pool per store |
| `store_id` | `STORE-07` | 20 stores total |
| `cashier_id` | `EMP-118` | 5–12 cashiers per store |
| `card_present` | `True` | Physical card tap/swipe |
| `cashier_override_flag` | `True` | Manager-key discount applied |

Online-only columns (`login_ip`, `device_fingerprint`, `session_duration_sec`,
`failed_logins_24h`, `vpn_proxy_flag`) are `None` for in-store rows.

## Example commands

```bash
# Clean labels, default channel split (35 % in-store)
python generate_loyalty_fraud_data.py --records 50000 --seed 42

# Dirty mode — medium noise, 40 % in-store
python generate_loyalty_fraud_data.py --records 50000 --mode dirty --dirt-level medium --channel-split 0.40 --seed 42

# Large clean dataset, skip slow feature engineering
python generate_loyalty_fraud_data.py --records 500000 --fraud-rate 0.03 --no-features --out large.csv

# Online-only (no in-store channel)
python generate_loyalty_fraud_data.py --records 10000 --channel-split 0 --seed 1

# Smoke test — fast, no features
python generate_loyalty_fraud_data.py --records 5000 --mode dirty --dirt-level medium --channel-split 0.35 --seed 42 --out test.csv --no-features
```

## Dependencies

Python 3.10+ with `numpy`, `pandas`, `faker` (no other runtime dependencies).
