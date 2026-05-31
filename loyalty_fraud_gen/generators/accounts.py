"""Account generation: legitimate accounts, referral rings, and in-store pools."""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
from faker import Faker

import numpy as np

from loyalty_fraud_gen.config import Config
from loyalty_fraud_gen.generators.base import BaseGenerator


class AccountGenerator(BaseGenerator):
    def __init__(self, rng: np.random.Generator, config: Config, fake: Faker) -> None:
        super().__init__(rng, config, fake)

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self, n_legit: int, n_rings: int
    ) -> tuple[pd.DataFrame, set, dict]:
        """Return (accounts_df, fraud_account_ids, store_pools).

        store_pools maps store_id → {terminals: [...], cashiers: [...]}
        """
        accounts:  list[dict] = []
        fraud_ids: set[int]   = set()
        acct_id = 1000

        acct_id = self._gen_legit(accounts, acct_id, n_legit)
        acct_id = self._gen_referral_rings(accounts, fraud_ids, acct_id, n_rings)

        df = pd.DataFrame(accounts)
        store_pools = self._build_store_pools()
        return df, fraud_ids, store_pools

    # ── Private helpers ───────────────────────────────────────────────────────

    def _gen_legit(self, accounts: list, acct_id: int, n: int) -> int:
        for _ in range(n):
            created = self.fake.date_time_between(start_date="-3y", end_date="-30d")
            email   = self.fake.email()
            accounts.append({
                "account_id":        acct_id,
                "email":             email,
                "email_domain":      email.split("@")[1],
                "account_age_days":  (datetime.now() - created).days,
                "created_at":        created,
                "country":           str(self.rng.choice(
                    ["IN"] * 8 + ["US", "GB", "AE", "SG"]
                )),
                "is_verified":       True,
                "referrer_id":       None,
                "account_type":      "legitimate",
                "baseline_spend":    float(self.rng.lognormal(mean=4.0, sigma=0.8)),
                "is_traveller":      bool(self.rng.random() < 0.06),
                "is_power_redeemer": bool(self.rng.random() < 0.04),
                "is_bulk_returner":  bool(self.rng.random() < 0.03),
                "is_gifting_buyer":  bool(self.rng.random() < 0.05),
            })
            acct_id += 1
        return acct_id

    def _gen_referral_rings(
        self, accounts: list, fraud_ids: set, acct_id: int, n_rings: int
    ) -> int:
        for _ in range(n_rings):
            ring_size      = int(self.rng.integers(3, 12))
            master_id      = acct_id
            created_master = self.fake.date_time_between(start_date="-6m", end_date="-7d")
            email          = self.fake.email()
            accounts.append({
                "account_id":        master_id,
                "email":             email,
                "email_domain":      email.split("@")[1],
                "account_age_days":  (datetime.now() - created_master).days,
                "created_at":        created_master,
                "country":           "IN",
                "is_verified":       True,
                "referrer_id":       None,
                "account_type":      "referral_fraud_master",
                "baseline_spend":    float(self.rng.lognormal(mean=3.5, sigma=0.5)),
                "is_traveller":      False,
                "is_power_redeemer": False,
                "is_bulk_returner":  False,
                "is_gifting_buyer":  False,
            })
            fraud_ids.add(master_id)
            acct_id += 1

            for j in range(ring_size):
                offset_h   = int(self.rng.integers(1, 72))
                created_sub = created_master + timedelta(hours=offset_h + j * 2)
                if self.rng.random() < 0.6:
                    domain    = str(self.rng.choice([
                        "tempmail.com", "throwam.com", "mailnull.com",
                        "yopmail.com", "guerrillamail.com",
                    ]))
                    sub_email = f"{self.fake.user_name()}{int(self.rng.integers(10,999))}@{domain}"
                else:
                    sub_email = self.fake.email()
                accounts.append({
                    "account_id":        acct_id,
                    "email":             sub_email,
                    "email_domain":      sub_email.split("@")[1],
                    "account_age_days":  (datetime.now() - created_sub).days,
                    "created_at":        created_sub,
                    "country":           "IN",
                    "is_verified":       bool(self.rng.random() < 0.4),
                    "referrer_id":       master_id,
                    "account_type":      "referral_fraud_sub",
                    "baseline_spend":    float(self.rng.lognormal(mean=2.5, sigma=0.5)),
                    "is_traveller":      False,
                    "is_power_redeemer": False,
                    "is_bulk_returner":  False,
                    "is_gifting_buyer":  False,
                })
                fraud_ids.add(acct_id)
                acct_id += 1

        return acct_id

    def _build_store_pools(self) -> dict:
        """Generate store/terminal/cashier pools for in-store channel."""
        cfg               = self.cfg
        pools             = {}
        terminal_counter  = 1
        cashier_counter   = 100

        for i in range(1, cfg.n_stores + 1):
            store_id = f"STORE-{i:02d}"
            n_terms  = int(self.rng.integers(
                cfg.min_terminals_per_store, cfg.max_terminals_per_store + 1
            ))
            n_cash   = int(self.rng.integers(
                cfg.min_cashiers_per_store, cfg.max_cashiers_per_store + 1
            ))
            pools[store_id] = {
                "terminals": [f"TERM-{terminal_counter + j:03d}" for j in range(n_terms)],
                "cashiers":  [f"EMP-{cashier_counter + j:03d}"  for j in range(n_cash)],
            }
            terminal_counter += n_terms
            cashier_counter  += n_cash

        return pools
