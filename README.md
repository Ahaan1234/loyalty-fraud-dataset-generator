# Loyalty Program Fraud Dataset Generator

Generates fully synthetic loyalty-program transaction data for training and benchmarking fraud-detection models. Produces two CSVs per run — a transactions file and an accounts file — with configurable fraud rates, data quality issues, and both online and in-store channels.

## Quick start

```bash
# Create and activate venv
python3 -m venv .venv && source .venv/bin/activate
pip install numpy pandas faker

# 50 k rows, clean labels, default settings
python generate_loyalty_fraud_data.py --records 50000 --seed 42

# 5 k rows fast smoke test (no feature engineering)
python generate_loyalty_fraud_data.py --records 5000 --mode dirty --dirt-level medium \
  --channel-split 0.35 --seed 42 --no-features
```

## Output files

| File | Description |
|------|-------------|
| `loyalty_fraud_data.csv` (default) | One row per transaction |
| `accounts_loyalty_fraud_data.csv` | One row per account |

Both paths follow `--out FILE` / `accounts_<FILE>`.

## Transaction schema

### Core columns (all rows)

| Column | Type | Notes |
|--------|------|-------|
| `txn_id` | int | Unique transaction identifier |
| `account_id` | int | Links to accounts file |
| `retailer` | str | One of 5 retailers |
| `timestamp` | datetime | UTC, past 365 days |
| `product_category` | str | groceries / electronics / clothing / homewares / beauty / sports / return |
| `amount_usd` | float | Negative for returns |
| `points_earned` | int | Base rate: 10 pts / $ |
| `points_redeemed` | int | Points spent on this transaction |
| `redemption_value_usd` | float | `points_redeemed × 0.01` |
| `is_return` | bool | True for refund transactions |
| `return_txn_id` | int / None | Original purchase txn_id |
| `new_device` | bool | First time this device seen on account |
| `referral_bonus_applied` | bool | Referral credit used |
| `fraud_type` | str | See fraud types below; `"none"` for legit |
| `is_fraud` | bool | Ground-truth label |
| `fp_risk_flag` | str | False-positive scenario tag (dirty mode) |
| `fn_risk_flag` | str | False-negative scenario tag (dirty mode) |
| `channel` | str | `"online"` or `"in_store"` |

### Online-only columns (None for in-store rows)

| Column | Type | Notes |
|--------|------|-------|
| `login_ip` | str | Domestic, foreign, or VPN IP |
| `device_fingerprint` | str | Browser / mobile device string |
| `session_duration_sec` | int | Seconds from login to checkout |
| `failed_logins_24h` | int | Failed attempts before this session |
| `vpn_proxy_flag` | bool | IP detected as VPN/proxy |

### In-store-only columns (None for online rows)

| Column | Type | Notes |
|--------|------|-------|
| `store_id` | str | e.g. `STORE-07` — pool of 20 stores |
| `pos_terminal_id` | str | e.g. `TERM-042` — 2–6 terminals per store |
| `cashier_id` | str | e.g. `EMP-118` — 5–12 cashiers per store |
| `card_present` | bool | Physical card tapped / swiped |
| `cashier_override_flag` | bool | Manager-key override applied |

### Dirty-mode-only column

| Column | Notes |
|--------|-------|
| `dirt_applied` | Pipe-delimited list of operations applied: `null_fields`, `ts_jitter`, `duplicate`, `pts_drift`, `label_noise` |

## Fraud types

### Online (`channel = "online"`)

| `fraud_type` | Description | Key signals |
|--------------|-------------|-------------|
| `account_takeover` | Credential-stuffed account; attacker drains points | Foreign IP, bot device, high `failed_logins_24h`, high redemption |
| `referral_fraud` | Fake accounts created solely to farm referral bonuses | Disposable email domain, purchase within minutes of account creation |
| `redemption_abuse` | Buy high-value item with points, return for cash | High `points_redeemed`, paired return within 14 days |

### In-store (`channel = "in_store"`)

| `fraud_type` | Description | Key signals |
|--------------|-------------|-------------|
| `card_id_swap` | Fraudster uses another person's physical loyalty card | Off-peak hour, store the account rarely visits, `card_present = True` |
| `employee_collusion` | Cashier awards phantom points to an accomplice with no real purchase | `amount_usd ≈ 0`, `points_earned` abnormally high, all transactions cluster on same `cashier_id`, timestamps at shift boundaries |
| `receipt_manipulation` | Fraudster submits a return with a forged or stolen receipt | `is_return = True`, `return_txn_id` belongs to a different account |
| `terminal_hopping` | Same account hits 4–8 distinct stores within 24–72 hours | High geographic velocity, consistent small redemptions across many `store_id` values |

### Edge cases (dirty mode only)

| Flag | Kind | Scenario |
|------|------|----------|
| `traveller_foreign_ip` | FP | Legit customer abroad — looks like ATO |
| `holiday_gifting_spree` | FP | Legit bulk buying — looks like account farming |
| `legit_bulk_return_purchase` / `_refund` | FP | Wrong-size return — looks like redemption abuse |
| `high_legit_redemption` | FP | Power loyalty member — high redemption ratio |
| `legitimate_multi_store_shopper` | FP | Works near multiple branches — looks like terminal hopping |
| `staff_discount_legitimate` | FP | Employee staff discount — looks like employee collusion |
| `slow_burn_ato` | FN | ATO attacker waits weeks, uses domestic IP — evades detection |
| `stealth_referral` | FN | Referral fraudster spends normally to blend in |
| `partial_return_abuse` | FN | Returns only 1 of 3 items — ratio stays below threshold |
| `collusion_spread_cashiers` | FN | Colluding cashiers rotate across terminals — signal stays thin |

## CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `--records N` | `50000` | Target row count (before dirtiness duplicates) |
| `--fraud-rate F` | `0.05` | Fraction of records labelled as fraud |
| `--seed S` | `42` | Random seed — fully reproducible output |
| `--out FILE` | `loyalty_fraud_data.csv` | Transactions output path |
| `--mode MODE` | `clean` | `clean` = perfect labels; `dirty` = FP/FN + noise |
| `--dirt-level LEVEL` | `medium` | `low` / `medium` / `high` — dirty mode only |
| `--channel-split F` | `0.35` | Fraction of records that are in-store (0–1) |
| `--no-features` | off | Skip rolling-window ML feature computation |

## Example commands

```bash
# Default: 50 k rows, clean, 35 % in-store
python generate_loyalty_fraud_data.py --records 50000 --seed 42

# Dirty mode, medium noise, 40 % in-store
python generate_loyalty_fraud_data.py --records 50000 --mode dirty --dirt-level medium \
  --channel-split 0.40 --seed 42

# Large dataset without slow feature engineering
python generate_loyalty_fraud_data.py --records 500000 --fraud-rate 0.03 \
  --no-features --out large.csv

# Online-only (no in-store transactions)
python generate_loyalty_fraud_data.py --records 10000 --channel-split 0 --seed 1

# High-noise dirty dataset for robustness testing
python generate_loyalty_fraud_data.py --records 50000 --mode dirty --dirt-level high \
  --fraud-rate 0.08 --seed 7
```

## Package structure

```
loyalty_fraud_gen/
├── config.py          # Constants + Config dataclass
├── models.py          # Transaction dataclass (replaces make_row)
├── generators/
│   ├── base.py        # BaseGenerator — RNG helpers shared by all generators
│   ├── accounts.py    # AccountGenerator — accounts DataFrame + store/terminal/cashier pools
│   ├── online.py      # OnlineFraudGenerator
│   └── instore.py     # InStoreFraudGenerator
├── pipeline/
│   ├── features.py    # DerivedFeatureEngine — per-account rolling-window features
│   └── dirtiness.py   # DirtinessLayer — null injection, jitter, duplicates, label noise
└── cli.py             # Argparse entrypoint + orchestration + summary report
generate_loyalty_fraud_data.py   # Backwards-compat shim → loyalty_fraud_gen.cli.main()
```

## Dependencies

Python 3.10+ · `numpy` · `pandas` · `faker`
