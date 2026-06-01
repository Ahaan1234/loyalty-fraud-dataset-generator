"""Online fraud generator: legit, ATO, referral fraud, redemption abuse, FP/FN."""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

from loyalty_fraud_gen.config import Config
from loyalty_fraud_gen.generators.base import BaseGenerator
from loyalty_fraud_gen.models import Transaction


class OnlineFraudGenerator(BaseGenerator):
    def __init__(self, rng: np.random.Generator, config: Config, fake: Faker) -> None:
        super().__init__(rng, config, fake)

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        accounts_df: pd.DataFrame,
        n_online: int,
        fraud_rate: float,
        mode: str,
        txn_id_start: int = 100_000,
    ) -> tuple[list[dict], int]:
        """Return (rows, next_available_txn_id)."""
        if n_online == 0:
            return [], txn_id_start

        rows:  list[dict] = []
        txn_id = txn_id_start
        start  = datetime.now() - timedelta(days=365)

        legit_accts = accounts_df[accounts_df["account_type"] == "legitimate"]
        ref_subs    = accounts_df[accounts_df["account_type"] == "referral_fraud_sub"]

        rs = int(self.rng.integers(0, 9999))
        ato_victims    = legit_accts.sample(
            n=max(1, int(len(legit_accts) * 0.04)), random_state=rs
        )
        rs = int(self.rng.integers(0, 9999))
        redemp_abusers = legit_accts.sample(
            n=max(1, int(len(legit_accts) * 0.06)), random_state=rs
        )

        n_fraud   = int(n_online * fraud_rate)
        n_ato     = int(n_fraud * 0.40)
        n_ref     = int(n_fraud * 0.35)
        n_redemp  = int(n_fraud * 0.25)
        n_legit   = n_online - n_fraud

        fp_budget = int(n_legit * 0.03) if mode == "dirty" else 0
        fn_budget = int(n_fraud * 0.08) if mode == "dirty" else 0

        # 1. Legitimate
        txn_id, legit_rows = self._gen_legit(legit_accts, n_legit, txn_id, start)
        rows.extend(legit_rows)

        # 2. ATO
        txn_id, ato_rows = self._gen_ato(ato_victims, n_ato, txn_id, start)
        rows.extend(ato_rows)

        # 3. Referral fraud
        if len(ref_subs) > 0 and n_ref > 0:
            txn_id, ref_rows = self._gen_referral(ref_subs, n_ref, txn_id)
            rows.extend(ref_rows)

        # 4. Redemption abuse
        txn_id, redemp_rows = self._gen_redemption_abuse(redemp_abusers, n_redemp, txn_id, start)
        rows.extend(redemp_rows)

        # 5. False Positives
        if fp_budget > 0:
            rs = int(self.rng.integers(0, 9999))
            fp_pool = legit_accts.sample(
                n=fp_budget, replace=True, random_state=rs
            ).reset_index(drop=True)
            txn_id, fp_rows = self._gen_fp(fp_pool, fp_budget, txn_id, start)
            rows.extend(fp_rows)

        # 6. False Negatives
        if fn_budget > 0:
            txn_id, fn_rows = self._gen_fn(
                ato_victims, ref_subs, redemp_abusers, fn_budget, txn_id, start
            )
            rows.extend(fn_rows)

        return rows, txn_id

    # ── Legitimate ────────────────────────────────────────────────────────────

    def _gen_legit(
        self, legit_accts: pd.DataFrame, n: int, txn_id: int, start: datetime
    ) -> tuple[int, list[dict]]:
        rows: list[dict] = []
        rs = int(self.rng.integers(0, 9999))
        sample = legit_accts.sample(n=n, replace=True, random_state=rs).reset_index(drop=True)
        cfg = self.cfg
        for i in range(n):
            acc  = sample.iloc[i]
            ts   = start + timedelta(seconds=int(self.rng.integers(0, 365 * 86400)))
            hour = self.weighted_choice(list(range(24)), cfg.legit_hour_w)
            ts   = ts.replace(hour=hour, minute=int(self.rng.integers(0, 60)))
            cat, amount, pts_e = self.sample_purchase(float(acc["baseline_spend"]))
            pts_r = int(self.rng.integers(50, 500)) if self.rng.random() < 0.12 else 0
            rows.append(Transaction(
                txn_id=txn_id,
                account_id=int(acc["account_id"]),
                retailer=str(self.rng.choice(cfg.retailers)),
                timestamp=ts,
                product_category=cat,
                amount_usd=amount,
                points_earned=pts_e,
                points_redeemed=pts_r,
                redemption_value_usd=round(pts_r * cfg.redemption_rate, 2),
                login_ip=self.random_ip("domestic"),
                device_fingerprint=self.random_device("legit"),
                is_return=False,
                return_txn_id=None,
                session_duration_sec=int(self.rng.integers(60, 1800)),
                failed_logins_24h=int(self.rng.integers(0, 2)),
                new_device=bool(self.rng.random() < 0.08),
                vpn_proxy_flag=bool(self.rng.random() < 0.03),
                referral_bonus_applied=False,
                fraud_type="none",
                is_fraud=False,
                channel="online",
            ).to_dict())
            txn_id += 1
        return txn_id, rows

    # ── Account Takeover ──────────────────────────────────────────────────────

    def _gen_ato(
        self, ato_victims: pd.DataFrame, n: int, txn_id: int, start: datetime
    ) -> tuple[int, list[dict]]:
        rows: list[dict] = []
        rs = int(self.rng.integers(0, 9999))
        sample = ato_victims.sample(n=n, replace=True, random_state=rs).reset_index(drop=True)
        cfg = self.cfg
        for i in range(n):
            acc          = sample.iloc[i]
            attack_start = start + timedelta(days=int(self.rng.integers(0, 360)))
            ts           = attack_start + timedelta(minutes=int(self.rng.integers(0, 180)))
            hour         = self.weighted_choice(list(range(24)), cfg.ato_hour_w)
            ts           = ts.replace(hour=hour, minute=int(self.rng.integers(0, 60)))
            cat, amount, pts_e = self.sample_purchase(float(acc["baseline_spend"]))
            pts_r = int(self.rng.integers(200, 5000))
            rows.append(Transaction(
                txn_id=txn_id,
                account_id=int(acc["account_id"]),
                retailer=str(self.rng.choice(cfg.retailers)),
                timestamp=ts,
                product_category=cat,
                amount_usd=amount,
                points_earned=pts_e,
                points_redeemed=pts_r,
                redemption_value_usd=round(pts_r * cfg.redemption_rate, 2),
                login_ip=self.random_ip("foreign"),
                device_fingerprint=self.random_device("bot"),
                is_return=False,
                return_txn_id=None,
                session_duration_sec=int(self.rng.integers(5, 90)),
                failed_logins_24h=int(self.rng.integers(3, 20)),
                new_device=True,
                vpn_proxy_flag=bool(self.rng.random() < 0.65),
                referral_bonus_applied=False,
                fraud_type="account_takeover",
                is_fraud=True,
                channel="online",
            ).to_dict())
            txn_id += 1
        return txn_id, rows

    # ── Referral Fraud ────────────────────────────────────────────────────────

    def _gen_referral(
        self, ref_subs: pd.DataFrame, n: int, txn_id: int
    ) -> tuple[int, list[dict]]:
        rows: list[dict] = []
        rs = int(self.rng.integers(0, 9999))
        sample = ref_subs.sample(n=n, replace=True, random_state=rs).reset_index(drop=True)
        cfg = self.cfg
        for i in range(len(sample)):
            acc       = sample.iloc[i]
            ts        = acc["created_at"] + timedelta(minutes=int(self.rng.integers(1, 30)))
            cat, _, _ = self.sample_purchase(float(acc["baseline_spend"]))
            amount    = round(float(self.rng.uniform(5, 30)), 2)
            pts_e     = int(amount * cfg.points_per_dollar)
            rows.append(Transaction(
                txn_id=txn_id,
                account_id=int(acc["account_id"]),
                retailer=str(self.rng.choice(cfg.retailers)),
                timestamp=ts,
                product_category=cat,
                amount_usd=amount,
                points_earned=pts_e,
                points_redeemed=0,
                redemption_value_usd=0.0,
                login_ip=self.random_ip("domestic"),
                device_fingerprint=self.random_device("legit"),
                is_return=False,
                return_txn_id=None,
                session_duration_sec=int(self.rng.integers(30, 300)),
                failed_logins_24h=0,
                new_device=bool(self.rng.random() < 0.5),
                vpn_proxy_flag=bool(self.rng.random() < 0.15),
                referral_bonus_applied=True,
                fraud_type="referral_fraud",
                is_fraud=True,
                channel="online",
            ).to_dict())
            txn_id += 1
        return txn_id, rows

    # ── Redemption Abuse ──────────────────────────────────────────────────────

    def _gen_redemption_abuse(
        self, redemp_abusers: pd.DataFrame, n: int, txn_id: int, start: datetime
    ) -> tuple[int, list[dict]]:
        rows: list[dict] = []
        rs = int(self.rng.integers(0, 9999))
        sample       = redemp_abusers.sample(n=n, replace=True, random_state=rs).reset_index(drop=True)
        return_pairs = []
        cfg          = self.cfg

        for i in range(n):
            acc  = sample.iloc[i]
            ts   = start + timedelta(days=int(self.rng.integers(30, 340)))
            hour = self.weighted_choice(list(range(24)), cfg.legit_hour_w)
            ts   = ts.replace(hour=hour)
            hi_cat = str(self.rng.choice(["electronics", "homewares", "clothing"]))
            lo, hi, _ = cfg.product_categories[hi_cat]
            amount = round(float(self.rng.uniform(hi * 0.6, hi)), 2)
            pts_e  = int(amount * cfg.points_per_dollar)
            pts_r  = int(amount * cfg.points_per_dollar * float(self.rng.uniform(0.5, 0.95)))
            rows.append(Transaction(
                txn_id=txn_id,
                account_id=int(acc["account_id"]),
                retailer=str(self.rng.choice(cfg.retailers)),
                timestamp=ts,
                product_category=hi_cat,
                amount_usd=amount,
                points_earned=pts_e,
                points_redeemed=pts_r,
                redemption_value_usd=round(pts_r * cfg.redemption_rate, 2),
                login_ip=self.random_ip("domestic"),
                device_fingerprint=self.random_device("legit"),
                is_return=False,
                return_txn_id=None,
                session_duration_sec=int(self.rng.integers(120, 900)),
                failed_logins_24h=0,
                new_device=bool(self.rng.random() < 0.05),
                vpn_proxy_flag=bool(self.rng.random() < 0.04),
                referral_bonus_applied=False,
                fraud_type="redemption_abuse",
                is_fraud=True,
                channel="online",
            ).to_dict())
            return_pairs.append((txn_id, int(acc["account_id"]), ts, amount, pts_r))
            txn_id += 1

        # Paired returns
        for (orig_id, acc_id, orig_ts, orig_amount, _) in return_pairs:
            ts_ret = orig_ts + timedelta(days=int(self.rng.integers(1, 14)))
            rows.append(Transaction(
                txn_id=txn_id,
                account_id=acc_id,
                retailer=str(self.rng.choice(cfg.retailers)),
                timestamp=ts_ret,
                product_category="return",
                amount_usd=-orig_amount,
                points_earned=-int(orig_amount * cfg.points_per_dollar),
                points_redeemed=0,
                redemption_value_usd=0.0,
                login_ip=self.random_ip("domestic"),
                device_fingerprint=self.random_device("legit"),
                is_return=True,
                return_txn_id=orig_id,
                session_duration_sec=int(self.rng.integers(60, 400)),
                failed_logins_24h=0,
                new_device=False,
                vpn_proxy_flag=False,
                referral_bonus_applied=False,
                fraud_type="redemption_abuse",
                is_fraud=True,
                channel="online",
            ).to_dict())
            txn_id += 1

        return txn_id, rows

    # ── False Positives ───────────────────────────────────────────────────────

    def _gen_fp(
        self, fp_pool: pd.DataFrame, fp_budget: int, txn_id: int, start: datetime
    ) -> tuple[int, list[dict]]:
        rows: list[dict] = []
        fp_types = [
            "traveller_foreign_ip", "holiday_gifting_spree",
            "legit_bulk_return",    "power_redeemer",
        ]
        cfg = self.cfg

        for i in range(fp_budget):
            acc     = fp_pool.iloc[i]
            fp_type = fp_types[i % len(fp_types)]

            if fp_type == "traveller_foreign_ip":
                ts   = start + timedelta(seconds=int(self.rng.integers(0, 365 * 86400)))
                hour = self.weighted_choice(list(range(24)), cfg.legit_hour_w)
                ts   = ts.replace(hour=hour, minute=int(self.rng.integers(0, 60)))
                cat, amount, pts_e = self.sample_purchase(float(acc["baseline_spend"]))
                pts_r = int(self.rng.integers(0, 200))
                rows.append(Transaction(
                    txn_id=txn_id,
                    account_id=int(acc["account_id"]),
                    retailer=str(self.rng.choice(cfg.retailers)),
                    timestamp=ts,
                    product_category=cat,
                    amount_usd=amount,
                    points_earned=pts_e,
                    points_redeemed=pts_r,
                    redemption_value_usd=round(pts_r * cfg.redemption_rate, 2),
                    login_ip=self.random_ip("foreign"),
                    device_fingerprint=self.random_device("legit"),
                    is_return=False,
                    return_txn_id=None,
                    session_duration_sec=int(self.rng.integers(300, 1800)),
                    failed_logins_24h=int(self.rng.integers(0, 1)),
                    new_device=True,
                    vpn_proxy_flag=bool(self.rng.random() < 0.3),
                    referral_bonus_applied=False,
                    fraud_type="none",
                    is_fraud=False,
                    fp_risk_flag="traveller_foreign_ip",
                    channel="online",
                ).to_dict())

            elif fp_type == "holiday_gifting_spree":
                base_ts = start + timedelta(days=int(self.rng.integers(300, 360)))
                n_spree = int(self.rng.integers(4, 9))
                for j in range(n_spree):
                    ts = base_ts + timedelta(
                        hours=j * 2 + int(self.rng.integers(0, 2))
                    )
                    cat, amount, pts_e = self.sample_purchase(
                        float(acc["baseline_spend"]) * 2
                    )
                    rows.append(Transaction(
                        txn_id=txn_id,
                        account_id=int(acc["account_id"]),
                        retailer=str(self.rng.choice(cfg.retailers)),
                        timestamp=ts,
                        product_category=cat,
                        amount_usd=amount,
                        points_earned=pts_e,
                        points_redeemed=0,
                        redemption_value_usd=0.0,
                        login_ip=self.random_ip("domestic"),
                        device_fingerprint=self.random_device("legit"),
                        is_return=False,
                        return_txn_id=None,
                        session_duration_sec=int(self.rng.integers(180, 900)),
                        failed_logins_24h=0,
                        new_device=False,
                        vpn_proxy_flag=False,
                        referral_bonus_applied=False,
                        fraud_type="none",
                        is_fraud=False,
                        fp_risk_flag="holiday_gifting_spree",
                        channel="online",
                    ).to_dict())
                    txn_id += 1

            elif fp_type == "legit_bulk_return":
                ts_buy = start + timedelta(days=int(self.rng.integers(10, 340)))
                cat    = str(self.rng.choice(["clothing", "electronics"]))
                lo, hi, _ = cfg.product_categories[cat]
                amount = round(float(self.rng.uniform(hi * 0.5, hi)), 2)
                pts_e  = int(amount * cfg.points_per_dollar)
                pts_r  = int(amount * cfg.points_per_dollar * float(self.rng.uniform(0.3, 0.6)))
                rows.append(Transaction(
                    txn_id=txn_id,
                    account_id=int(acc["account_id"]),
                    retailer=str(self.rng.choice(cfg.retailers)),
                    timestamp=ts_buy,
                    product_category=cat,
                    amount_usd=amount,
                    points_earned=pts_e,
                    points_redeemed=pts_r,
                    redemption_value_usd=round(pts_r * cfg.redemption_rate, 2),
                    login_ip=self.random_ip("domestic"),
                    device_fingerprint=self.random_device("legit"),
                    is_return=False,
                    return_txn_id=None,
                    session_duration_sec=int(self.rng.integers(300, 1200)),
                    failed_logins_24h=0,
                    new_device=False,
                    vpn_proxy_flag=False,
                    referral_bonus_applied=False,
                    fraud_type="none",
                    is_fraud=False,
                    fp_risk_flag="legit_bulk_return_purchase",
                    channel="online",
                ).to_dict())
                orig_id = txn_id
                txn_id += 1
                ts_ret = ts_buy + timedelta(days=int(self.rng.integers(2, 10)))
                rows.append(Transaction(
                    txn_id=txn_id,
                    account_id=int(acc["account_id"]),
                    retailer=str(self.rng.choice(cfg.retailers)),
                    timestamp=ts_ret,
                    product_category="return",
                    amount_usd=-amount,
                    points_earned=-pts_e,
                    points_redeemed=0,
                    redemption_value_usd=0.0,
                    login_ip=self.random_ip("domestic"),
                    device_fingerprint=self.random_device("legit"),
                    is_return=True,
                    return_txn_id=orig_id,
                    session_duration_sec=int(self.rng.integers(120, 600)),
                    failed_logins_24h=0,
                    new_device=False,
                    vpn_proxy_flag=False,
                    referral_bonus_applied=False,
                    fraud_type="none",
                    is_fraud=False,
                    fp_risk_flag="legit_bulk_return_refund",
                    channel="online",
                ).to_dict())

            elif fp_type == "power_redeemer":
                ts = start + timedelta(seconds=int(self.rng.integers(0, 365 * 86400)))
                cat, amount, pts_e = self.sample_purchase(float(acc["baseline_spend"]))
                pts_r = int(pts_e * float(self.rng.uniform(0.8, 1.2)))
                rows.append(Transaction(
                    txn_id=txn_id,
                    account_id=int(acc["account_id"]),
                    retailer=str(self.rng.choice(cfg.retailers)),
                    timestamp=ts,
                    product_category=cat,
                    amount_usd=amount,
                    points_earned=pts_e,
                    points_redeemed=pts_r,
                    redemption_value_usd=round(pts_r * cfg.redemption_rate, 2),
                    login_ip=self.random_ip("domestic"),
                    device_fingerprint=self.random_device("legit"),
                    is_return=False,
                    return_txn_id=None,
                    session_duration_sec=int(self.rng.integers(200, 1500)),
                    failed_logins_24h=0,
                    new_device=False,
                    vpn_proxy_flag=False,
                    referral_bonus_applied=False,
                    fraud_type="none",
                    is_fraud=False,
                    fp_risk_flag="high_legit_redemption",
                    channel="online",
                ).to_dict())

            txn_id += 1

        return txn_id, rows

    # ── False Negatives ───────────────────────────────────────────────────────

    def _gen_fn(
        self,
        ato_victims:    pd.DataFrame,
        ref_subs:       pd.DataFrame,
        redemp_abusers: pd.DataFrame,
        fn_budget:      int,
        txn_id:         int,
        start:          datetime,
    ) -> tuple[int, list[dict]]:
        rows: list[dict] = []
        cfg = self.cfg

        rs = int(self.rng.integers(0, 9999))
        fn_victims = ato_victims.sample(
            n=max(1, fn_budget // 3), replace=True, random_state=rs
        ).reset_index(drop=True)

        rs = int(self.rng.integers(0, 9999))
        fn_ref_pool = (
            ref_subs.sample(n=max(1, fn_budget // 3), replace=True, random_state=rs).reset_index(drop=True)
            if len(ref_subs) > 0
            else pd.DataFrame()
        )

        rs = int(self.rng.integers(0, 9999))
        fn_redemp_pool = redemp_abusers.sample(
            n=max(1, fn_budget // 3), replace=True, random_state=rs
        ).reset_index(drop=True)

        fn_types_assigned = (
            [("slow_burn_ato",        fn_victims,     i) for i in range(len(fn_victims))]
            + ([("stealth_referral",  fn_ref_pool,    i) for i in range(len(fn_ref_pool))]
               if len(fn_ref_pool) > 0 else [])
            + [("partial_return_abuse", fn_redemp_pool, i) for i in range(len(fn_redemp_pool))]
        )

        for (fn_type, pool, idx) in fn_types_assigned[:fn_budget]:
            if idx >= len(pool):
                continue
            acc = pool.iloc[idx]

            if fn_type == "slow_burn_ato":
                ts   = start + timedelta(days=int(self.rng.integers(30, 300)))
                hour = self.weighted_choice(list(range(24)), cfg.legit_hour_w)
                ts   = ts.replace(hour=hour, minute=int(self.rng.integers(0, 60)))
                cat, amount, pts_e = self.sample_purchase(float(acc["baseline_spend"]))
                pts_r = int(self.rng.integers(500, 3000))
                rows.append(Transaction(
                    txn_id=txn_id,
                    account_id=int(acc["account_id"]),
                    retailer=str(self.rng.choice(cfg.retailers)),
                    timestamp=ts,
                    product_category=cat,
                    amount_usd=amount,
                    points_earned=pts_e,
                    points_redeemed=pts_r,
                    redemption_value_usd=round(pts_r * cfg.redemption_rate, 2),
                    login_ip=self.random_ip("domestic"),
                    device_fingerprint=self.random_device("legit"),
                    is_return=False,
                    return_txn_id=None,
                    session_duration_sec=int(self.rng.integers(200, 900)),
                    failed_logins_24h=int(self.rng.integers(0, 2)),
                    new_device=False,
                    vpn_proxy_flag=False,
                    referral_bonus_applied=False,
                    fraud_type="account_takeover",
                    is_fraud=True,
                    fn_risk_flag="slow_burn_ato",
                    channel="online",
                ).to_dict())

            elif fn_type == "stealth_referral":
                acc_created = acc.get("created_at", start + timedelta(days=30))
                ts = acc_created + timedelta(days=int(self.rng.integers(3, 14)))
                cat, amount, pts_e = self.sample_purchase(60.0)
                rows.append(Transaction(
                    txn_id=txn_id,
                    account_id=int(acc["account_id"]),
                    retailer=str(self.rng.choice(cfg.retailers)),
                    timestamp=ts,
                    product_category=cat,
                    amount_usd=amount,
                    points_earned=pts_e,
                    points_redeemed=0,
                    redemption_value_usd=0.0,
                    login_ip=self.random_ip("domestic"),
                    device_fingerprint=self.random_device("legit"),
                    is_return=False,
                    return_txn_id=None,
                    session_duration_sec=int(self.rng.integers(300, 1200)),
                    failed_logins_24h=0,
                    new_device=False,
                    vpn_proxy_flag=False,
                    referral_bonus_applied=True,
                    fraud_type="referral_fraud",
                    is_fraud=True,
                    fn_risk_flag="stealth_referral",
                    channel="online",
                ).to_dict())

            elif fn_type == "partial_return_abuse":
                ts_buy = start + timedelta(days=int(self.rng.integers(30, 300)))
                cat    = str(self.rng.choice(["electronics", "homewares"]))
                lo, hi, _ = cfg.product_categories[cat]
                amount = round(float(self.rng.uniform(hi * 0.4, hi * 0.7)), 2)
                pts_e  = int(amount * cfg.points_per_dollar)
                pts_r  = int(amount * cfg.points_per_dollar * float(self.rng.uniform(0.4, 0.65)))
                rows.append(Transaction(
                    txn_id=txn_id,
                    account_id=int(acc["account_id"]),
                    retailer=str(self.rng.choice(cfg.retailers)),
                    timestamp=ts_buy,
                    product_category=cat,
                    amount_usd=amount,
                    points_earned=pts_e,
                    points_redeemed=pts_r,
                    redemption_value_usd=round(pts_r * cfg.redemption_rate, 2),
                    login_ip=self.random_ip("domestic"),
                    device_fingerprint=self.random_device("legit"),
                    is_return=False,
                    return_txn_id=None,
                    session_duration_sec=int(self.rng.integers(200, 800)),
                    failed_logins_24h=0,
                    new_device=False,
                    vpn_proxy_flag=False,
                    referral_bonus_applied=False,
                    fraud_type="redemption_abuse",
                    is_fraud=True,
                    fn_risk_flag="partial_return_abuse",
                    channel="online",
                ).to_dict())
                orig_id = txn_id
                txn_id += 1
                partial_amount = round(amount / 3, 2)
                ts_ret = ts_buy + timedelta(days=int(self.rng.integers(5, 20)))
                rows.append(Transaction(
                    txn_id=txn_id,
                    account_id=int(acc["account_id"]),
                    retailer=str(self.rng.choice(cfg.retailers)),
                    timestamp=ts_ret,
                    product_category="return",
                    amount_usd=-partial_amount,
                    points_earned=-int(partial_amount * cfg.points_per_dollar),
                    points_redeemed=0,
                    redemption_value_usd=0.0,
                    login_ip=self.random_ip("domestic"),
                    device_fingerprint=self.random_device("legit"),
                    is_return=True,
                    return_txn_id=orig_id,
                    session_duration_sec=int(self.rng.integers(60, 300)),
                    failed_logins_24h=0,
                    new_device=False,
                    vpn_proxy_flag=False,
                    referral_bonus_applied=False,
                    fraud_type="redemption_abuse",
                    is_fraud=True,
                    fn_risk_flag="partial_return_abuse",
                    channel="online",
                ).to_dict())

            txn_id += 1

        return txn_id, rows
