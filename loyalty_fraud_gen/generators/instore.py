"""In-store fraud generator: legit in-store + 4 new fraud types + in-store FP/FN."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from faker import Faker

from loyalty_fraud_gen.config import Config
from loyalty_fraud_gen.generators.base import BaseGenerator
from loyalty_fraud_gen.models import Transaction


class InStoreFraudGenerator(BaseGenerator):
    def __init__(self, rng: np.random.Generator, config: Config, fake: Faker) -> None:
        super().__init__(rng, config, fake)

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        accounts_df:    pd.DataFrame,
        n_instore:      int,
        fraud_rate:     float,
        mode:           str,
        store_pools:    dict,
        txn_id_start:   int = 200_000,
        existing_txns:  Optional[list[dict]] = None,
    ) -> tuple[list[dict], int]:
        """Return (rows, next_available_txn_id).

        existing_txns — list of online transaction dicts; used by
        receipt_manipulation to pick a realistic stolen receipt txn_id.
        """
        if n_instore == 0:
            return [], txn_id_start

        rows:  list[dict] = []
        txn_id = txn_id_start
        start  = datetime.now() - timedelta(days=365)

        legit_accts = accounts_df[accounts_df["account_type"] == "legitimate"]
        stores      = list(store_pools.keys())

        n_fraud  = int(n_instore * fraud_rate)
        n_legit  = n_instore - n_fraud

        fp_budget = int(n_legit * 0.03) if mode == "dirty" else 0
        fn_budget = int(n_fraud * 0.08) if mode == "dirty" else 0

        # Budget per fraud type (equal split across 4 types)
        budget_each = max(1, n_fraud // 4)

        # 1. Legitimate in-store
        txn_id, legit_rows = self._gen_legit(legit_accts, n_legit, txn_id, start, store_pools, stores)
        rows.extend(legit_rows)

        # 2. card_id_swap
        txn_id, cs_rows = self._gen_card_id_swap(legit_accts, budget_each, txn_id, start, store_pools, stores)
        rows.extend(cs_rows)

        # 3. employee_collusion
        txn_id, col_rows = self._gen_employee_collusion(
            legit_accts, budget_each, txn_id, start, store_pools, stores
        )
        rows.extend(col_rows)

        # 4. receipt_manipulation
        txn_id, rm_rows = self._gen_receipt_manipulation(
            legit_accts, budget_each, txn_id, start, store_pools, stores, existing_txns
        )
        rows.extend(rm_rows)

        # 5. terminal_hopping
        txn_id, th_rows = self._gen_terminal_hopping(
            legit_accts, budget_each, txn_id, start, store_pools, stores
        )
        rows.extend(th_rows)

        # 6. FP edge cases (dirty mode only)
        if fp_budget > 0:
            txn_id, fp_rows = self._gen_fp(
                legit_accts, fp_budget, txn_id, start, store_pools, stores
            )
            rows.extend(fp_rows)

        # 7. FN edge cases (dirty mode only)
        if fn_budget > 0:
            txn_id, fn_rows = self._gen_fn(
                legit_accts, fn_budget, txn_id, start, store_pools, stores
            )
            rows.extend(fn_rows)

        return rows, txn_id

    # ── Shared builder ────────────────────────────────────────────────────────

    def _make_instore(
        self,
        txn_id:               int,
        account_id:           int,
        ts:                   datetime,
        store_id:             str,
        terminal_id:          str,
        cashier_id:           str,
        cat:                  str,
        amount:               float,
        pts_e:                int,
        pts_r:                int = 0,
        is_return:            bool = False,
        return_txn_id:        Optional[int] = None,
        card_present:         bool = True,
        cashier_override_flag: bool = False,
        fraud_type:           str = "none",
        is_fraud:             bool = False,
        fp_risk_flag:         str = "none",
        fn_risk_flag:         str = "none",
    ) -> dict:
        cfg = self.cfg
        return Transaction(
            txn_id=txn_id,
            account_id=account_id,
            retailer=str(self.rng.choice(cfg.retailers)),
            timestamp=ts,
            product_category=cat,
            amount_usd=amount,
            points_earned=pts_e,
            points_redeemed=pts_r,
            redemption_value_usd=round(pts_r * cfg.redemption_rate, 2),
            login_ip=None,
            device_fingerprint=None,
            is_return=is_return,
            return_txn_id=return_txn_id,
            session_duration_sec=None,
            failed_logins_24h=None,
            new_device=False,
            vpn_proxy_flag=None,
            referral_bonus_applied=False,
            fraud_type=fraud_type,
            is_fraud=is_fraud,
            fp_risk_flag=fp_risk_flag,
            fn_risk_flag=fn_risk_flag,
            channel="in_store",
            pos_terminal_id=terminal_id,
            store_id=store_id,
            cashier_id=cashier_id,
            card_present=card_present,
            cashier_override_flag=cashier_override_flag,
        ).to_dict()

    def _pick_store_context(self, store_pools: dict, store_id: str) -> tuple[str, str]:
        """Return (terminal_id, cashier_id) for the given store."""
        pool       = store_pools[store_id]
        terminal   = str(self.rng.choice(pool["terminals"]))
        cashier    = str(self.rng.choice(pool["cashiers"]))
        return terminal, cashier

    # ── 1. Legitimate in-store ────────────────────────────────────────────────

    def _gen_legit(
        self,
        legit_accts: pd.DataFrame,
        n:           int,
        txn_id:      int,
        start:       datetime,
        store_pools: dict,
        stores:      list,
    ) -> tuple[int, list[dict]]:
        rows: list[dict] = []
        rs = int(self.rng.integers(0, 9999))
        sample = legit_accts.sample(n=n, replace=True, random_state=rs).reset_index(drop=True)
        cfg = self.cfg

        for i in range(n):
            acc      = sample.iloc[i]
            ts       = start + timedelta(seconds=int(self.rng.integers(0, 365 * 86400)))
            hour     = self.weighted_choice(list(range(24)), cfg.legit_hour_w)
            ts       = ts.replace(hour=hour, minute=int(self.rng.integers(0, 60)))
            store_id = str(self.rng.choice(stores))
            term, emp = self._pick_store_context(store_pools, store_id)
            cat, amount, pts_e = self.sample_purchase(float(acc["baseline_spend"]))
            pts_r = int(self.rng.integers(50, 500)) if self.rng.random() < 0.10 else 0
            rows.append(self._make_instore(
                txn_id=txn_id,
                account_id=int(acc["account_id"]),
                ts=ts, store_id=store_id, terminal_id=term, cashier_id=emp,
                cat=cat, amount=amount, pts_e=pts_e, pts_r=pts_r,
                card_present=bool(self.rng.random() < 0.85),
            ))
            txn_id += 1

        return txn_id, rows

    # ── 2. card_id_swap ───────────────────────────────────────────────────────

    def _gen_card_id_swap(
        self,
        legit_accts: pd.DataFrame,
        n:           int,
        txn_id:      int,
        start:       datetime,
        store_pools: dict,
        stores:      list,
    ) -> tuple[int, list[dict]]:
        """Fraudster uses another person's physical loyalty card at the till.

        Signal: unfamiliar store + off-peak hour.
        """
        rows: list[dict] = []
        rs = int(self.rng.integers(0, 9999))
        sample = legit_accts.sample(n=n, replace=True, random_state=rs).reset_index(drop=True)
        cfg    = self.cfg
        # Use the second half of stores to simulate "unfamiliar" locations
        unfamiliar = stores[len(stores) // 2:]
        if not unfamiliar:
            unfamiliar = stores

        for i in range(n):
            acc      = sample.iloc[i]
            ts       = start + timedelta(seconds=int(self.rng.integers(0, 365 * 86400)))
            hour     = self.weighted_choice(list(range(24)), cfg.offpeak_hour_w)
            ts       = ts.replace(hour=hour, minute=int(self.rng.integers(0, 60)))
            store_id = str(self.rng.choice(unfamiliar))
            term, emp = self._pick_store_context(store_pools, store_id)
            cat, amount, pts_e = self.sample_purchase(float(acc["baseline_spend"]))
            rows.append(self._make_instore(
                txn_id=txn_id,
                account_id=int(acc["account_id"]),
                ts=ts, store_id=store_id, terminal_id=term, cashier_id=emp,
                cat=cat, amount=amount, pts_e=pts_e,
                card_present=True,
                cashier_override_flag=False,
                fraud_type="card_id_swap",
                is_fraud=True,
            ))
            txn_id += 1

        return txn_id, rows

    # ── 3. employee_collusion ─────────────────────────────────────────────────

    def _gen_employee_collusion(
        self,
        legit_accts: pd.DataFrame,
        n_target:    int,
        txn_id:      int,
        start:       datetime,
        store_pools: dict,
        stores:      list,
    ) -> tuple[int, list[dict]]:
        """Cashier awards phantom points to accomplice accounts (0-value transactions).

        Signal: zero-amount + high points_earned, all on same cashier_id.
        Timestamps cluster at shift boundaries (6-7 am / 9-10 pm).
        """
        rows: list[dict] = []
        cfg = self.cfg

        # Shift-boundary hours: 60 % morning start, 40 % evening end
        shift_hours_am = [6, 7]
        shift_hours_pm = [21, 22]

        generated = 0
        while generated < n_target:
            # Pick one colluding cashier at a random store
            store_id  = str(self.rng.choice(stores))
            pool      = store_pools[store_id]
            cashier   = str(self.rng.choice(pool["cashiers"]))
            terminal  = str(self.rng.choice(pool["terminals"]))

            n_accomplices = int(self.rng.integers(3, 9))
            rs = int(self.rng.integers(0, 9999))
            accomplices = legit_accts.sample(
                n=min(n_accomplices, len(legit_accts)), replace=True, random_state=rs
            ).reset_index(drop=True)

            for j in range(len(accomplices)):
                if generated >= n_target:
                    break
                acc = accomplices.iloc[j]

                # Cluster timestamps around shift start or end
                base_day = start + timedelta(days=int(self.rng.integers(0, 360)))
                if self.rng.random() < 0.6:
                    hour = int(self.rng.choice(shift_hours_am))
                else:
                    hour = int(self.rng.choice(shift_hours_pm))
                ts = base_day.replace(hour=hour, minute=int(self.rng.integers(0, 20)))

                # Phantom transaction: near-zero amount, high points
                amount   = round(float(self.rng.uniform(0, 2)), 2)
                pts_e    = int(self.rng.integers(500, 2001))
                override = bool(self.rng.random() < 0.80)

                rows.append(self._make_instore(
                    txn_id=txn_id,
                    account_id=int(acc["account_id"]),
                    ts=ts,
                    store_id=store_id,
                    terminal_id=terminal,
                    cashier_id=cashier,
                    cat="groceries",
                    amount=amount,
                    pts_e=pts_e,
                    card_present=True,
                    cashier_override_flag=override,
                    fraud_type="employee_collusion",
                    is_fraud=True,
                ))
                txn_id  += 1
                generated += 1

        return txn_id, rows

    # ── 4. receipt_manipulation ───────────────────────────────────────────────

    def _gen_receipt_manipulation(
        self,
        legit_accts:   pd.DataFrame,
        n:             int,
        txn_id:        int,
        start:         datetime,
        store_pools:   dict,
        stores:        list,
        existing_txns: Optional[list[dict]],
    ) -> tuple[int, list[dict]]:
        """Fraudster submits a return using a forged / stolen receipt.

        Signal: return_txn_id belongs to a different account_id; no matching
        purchase exists in this account's own history.
        """
        rows: list[dict] = []
        rs = int(self.rng.integers(0, 9999))
        sample = legit_accts.sample(n=n, replace=True, random_state=rs).reset_index(drop=True)
        cfg    = self.cfg

        # Build a lookup of (txn_id → account_id) from existing online rows
        # so we can pick a txn_id that genuinely belongs to a different account.
        online_pool: list[tuple[int, int]] = []   # [(txn_id, account_id)]
        if existing_txns:
            for row in existing_txns:
                if not row.get("is_fraud", False) and not row.get("is_return", False):
                    online_pool.append((int(row["txn_id"]), int(row["account_id"])))

        for i in range(n):
            acc   = sample.iloc[i]
            acc_id = int(acc["account_id"])
            ts    = start + timedelta(seconds=int(self.rng.integers(0, 365 * 86400)))
            hour  = self.weighted_choice(list(range(24)), cfg.legit_hour_w)
            ts    = ts.replace(hour=hour, minute=int(self.rng.integers(0, 60)))

            store_id  = str(self.rng.choice(stores))
            term, emp = self._pick_store_context(store_pools, store_id)

            # Stolen receipt references a different account's transaction
            stolen_txn_id: Optional[int] = None
            if online_pool:
                candidates = [t for t in online_pool if t[1] != acc_id]
                if candidates:
                    victim_txn, _ = candidates[int(self.rng.integers(0, len(candidates)))]
                    stolen_txn_id = victim_txn

            if stolen_txn_id is None:
                # Fallback: plausible ID in the online range that's not this account
                stolen_txn_id = int(self.rng.integers(100_000, 150_000))

            hi_cat = str(self.rng.choice(["electronics", "clothing", "homewares"]))
            lo, hi, _ = cfg.product_categories[hi_cat]
            refund_amount = round(float(self.rng.uniform(lo * 2, hi * 0.7)), 2)
            pts_clawback  = int(refund_amount * cfg.points_per_dollar)

            rows.append(self._make_instore(
                txn_id=txn_id,
                account_id=acc_id,
                ts=ts,
                store_id=store_id,
                terminal_id=term,
                cashier_id=emp,
                cat="return",
                amount=-refund_amount,
                pts_e=-pts_clawback,
                is_return=True,
                return_txn_id=stolen_txn_id,
                card_present=True,
                cashier_override_flag=False,
                fraud_type="receipt_manipulation",
                is_fraud=True,
            ))
            txn_id += 1

        return txn_id, rows

    # ── 5. terminal_hopping ───────────────────────────────────────────────────

    def _gen_terminal_hopping(
        self,
        legit_accts: pd.DataFrame,
        n_target:    int,
        txn_id:      int,
        start:       datetime,
        store_pools: dict,
        stores:      list,
    ) -> tuple[int, list[dict]]:
        """Same account hits multiple stores in a tight 24-72 hour window.

        Signal: geographic velocity — many distinct stores, tight time window.
        """
        rows: list[dict] = []
        cfg = self.cfg

        generated = 0
        while generated < n_target:
            rs = int(self.rng.integers(0, 9999))
            acc = legit_accts.sample(n=1, random_state=rs).iloc[0]
            acc_id = int(acc["account_id"])

            n_hops    = int(self.rng.integers(4, 9))
            hop_hours = int(self.rng.integers(24, 73))  # spread over 24-72 hours
            base_ts   = start + timedelta(days=int(self.rng.integers(0, 355)))

            # Pick n_hops distinct stores (sample without replacement if possible)
            if len(stores) >= n_hops:
                hop_stores = list(self.rng.choice(stores, size=n_hops, replace=False))
            else:
                hop_stores = list(self.rng.choice(stores, size=n_hops, replace=True))

            for k in range(n_hops):
                if generated >= n_target:
                    break
                # Distribute hops evenly across the time window
                offset_h   = int(k * hop_hours / n_hops) + int(self.rng.integers(0, 2))
                ts         = base_ts + timedelta(hours=offset_h)
                store_id   = hop_stores[k]
                term, emp  = self._pick_store_context(store_pools, store_id)

                # Small-to-medium redemption
                cat, amount, pts_e = self.sample_purchase(float(acc["baseline_spend"]))
                pts_r = int(self.rng.integers(100, 800))

                rows.append(self._make_instore(
                    txn_id=txn_id,
                    account_id=acc_id,
                    ts=ts,
                    store_id=store_id,
                    terminal_id=term,
                    cashier_id=emp,
                    cat=cat,
                    amount=amount,
                    pts_e=pts_e,
                    pts_r=pts_r,
                    card_present=True,
                    cashier_override_flag=False,
                    fraud_type="terminal_hopping",
                    is_fraud=True,
                ))
                txn_id    += 1
                generated += 1

        return txn_id, rows

    # ── 6. FP edge cases ──────────────────────────────────────────────────────

    def _gen_fp(
        self,
        legit_accts: pd.DataFrame,
        fp_budget:   int,
        txn_id:      int,
        start:       datetime,
        store_pools: dict,
        stores:      list,
    ) -> tuple[int, list[dict]]:
        """In-store false positives (dirty mode only)."""
        rows: list[dict] = []
        cfg    = self.cfg
        fp_types = [
            "legitimate_multi_store_shopper",
            "staff_discount_legitimate",
        ]

        for i in range(fp_budget):
            rs  = int(self.rng.integers(0, 9999))
            acc = legit_accts.sample(n=1, random_state=rs).iloc[0]
            acc_id  = int(acc["account_id"])
            fp_type = fp_types[i % len(fp_types)]

            if fp_type == "legitimate_multi_store_shopper":
                # Legit customer who genuinely shops at many branches
                # (looks like terminal_hopping but is_fraud=False)
                n_visits = int(self.rng.integers(6, 11))
                base_ts  = start + timedelta(days=int(self.rng.integers(0, 355)))
                if len(stores) >= n_visits:
                    visit_stores = list(self.rng.choice(stores, size=n_visits, replace=False))
                else:
                    visit_stores = list(self.rng.choice(stores, size=n_visits, replace=True))
                for k in range(n_visits):
                    ts        = base_ts + timedelta(days=k, hours=int(self.rng.integers(8, 20)))
                    store_id  = visit_stores[k]
                    term, emp = self._pick_store_context(store_pools, store_id)
                    cat, amount, pts_e = self.sample_purchase(float(acc["baseline_spend"]))
                    rows.append(self._make_instore(
                        txn_id=txn_id,
                        account_id=acc_id,
                        ts=ts,
                        store_id=store_id,
                        terminal_id=term,
                        cashier_id=emp,
                        cat=cat,
                        amount=amount,
                        pts_e=pts_e,
                        card_present=True,
                        fraud_type="none",
                        is_fraud=False,
                        fp_risk_flag="legitimate_multi_store_shopper",
                    ))
                    txn_id += 1

            elif fp_type == "staff_discount_legitimate":
                # Employee using their own staff discount: triggers collusion signals
                ts       = start + timedelta(seconds=int(self.rng.integers(0, 365 * 86400)))
                store_id = str(self.rng.choice(stores))
                term, emp = self._pick_store_context(store_pools, store_id)
                cat, amount, pts_e = self.sample_purchase(float(acc["baseline_spend"]))
                # Small/zero amount (staff gets it free/discounted)
                disc_amount = round(amount * float(self.rng.uniform(0, 0.3)), 2)
                disc_pts    = int(disc_amount * cfg.points_per_dollar)
                rows.append(self._make_instore(
                    txn_id=txn_id,
                    account_id=acc_id,
                    ts=ts,
                    store_id=store_id,
                    terminal_id=term,
                    cashier_id=emp,
                    cat=cat,
                    amount=disc_amount,
                    pts_e=disc_pts,
                    card_present=True,
                    cashier_override_flag=True,  # override used for staff discount
                    fraud_type="none",
                    is_fraud=False,
                    fp_risk_flag="staff_discount_legitimate",
                ))
                txn_id += 1

        return txn_id, rows

    # ── 7. FN edge cases ──────────────────────────────────────────────────────

    def _gen_fn(
        self,
        legit_accts: pd.DataFrame,
        fn_budget:   int,
        txn_id:      int,
        start:       datetime,
        store_pools: dict,
        stores:      list,
    ) -> tuple[int, list[dict]]:
        """In-store false negatives (dirty mode only): fraud that evades detection.

        collusion_spread_cashiers — colluding cashiers rotate across terminals
        so no single cashier accumulates enough signal.
        """
        rows: list[dict] = []
        cfg = self.cfg

        generated = 0
        while generated < fn_budget:
            # Pick a set of colluding cashiers from different stores
            n_cashiers = int(self.rng.integers(2, 5))
            colluding: list[tuple[str, str, str]] = []  # [(store, terminal, cashier)]
            for _ in range(n_cashiers):
                sid = str(self.rng.choice(stores))
                t, c = self._pick_store_context(store_pools, sid)
                colluding.append((sid, t, c))

            rs = int(self.rng.integers(0, 9999))
            accomplices = legit_accts.sample(
                n=min(int(self.rng.integers(2, 6)), len(legit_accts)),
                replace=True,
                random_state=rs,
            ).reset_index(drop=True)

            base_day = start + timedelta(days=int(self.rng.integers(0, 355)))

            for j, acc in accomplices.iterrows():
                if generated >= fn_budget:
                    break
                # Rotate through cashiers to spread signal
                store_id, terminal, cashier = colluding[j % len(colluding)]
                ts     = base_day + timedelta(
                    hours=int(self.rng.integers(0, 48)),
                    minutes=int(self.rng.integers(0, 60)),
                )
                amount = round(float(self.rng.uniform(0, 3)), 2)
                pts_e  = int(self.rng.integers(300, 800))  # high but not extreme

                rows.append(self._make_instore(
                    txn_id=txn_id,
                    account_id=int(acc["account_id"]),
                    ts=ts,
                    store_id=store_id,
                    terminal_id=terminal,
                    cashier_id=cashier,
                    cat="groceries",
                    amount=amount,
                    pts_e=pts_e,
                    card_present=True,
                    cashier_override_flag=bool(self.rng.random() < 0.5),
                    fraud_type="employee_collusion",
                    is_fraud=True,
                    fn_risk_flag="collusion_spread_cashiers",
                ))
                txn_id    += 1
                generated += 1

        return txn_id, rows
