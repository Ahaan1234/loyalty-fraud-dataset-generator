"""Core dataclasses for the loyalty fraud generator."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional


@dataclass
class Transaction:
    """Single loyalty-program transaction row.

    Online-only columns (None for in_store channel):
        login_ip, device_fingerprint, session_duration_sec,
        failed_logins_24h, vpn_proxy_flag

    In-store-only columns (None for online channel):
        pos_terminal_id, store_id, cashier_id,
        card_present, cashier_override_flag
    """
    txn_id:                  int
    account_id:              int
    retailer:                str
    timestamp:               datetime
    product_category:        str
    amount_usd:              float
    points_earned:           int
    points_redeemed:         int
    redemption_value_usd:    float
    login_ip:                Optional[str]
    device_fingerprint:      Optional[str]
    is_return:               bool
    return_txn_id:           Optional[int]
    session_duration_sec:    Optional[int]
    failed_logins_24h:       Optional[int]
    new_device:              bool
    vpn_proxy_flag:          Optional[bool]
    referral_bonus_applied:  bool
    fraud_type:              str
    is_fraud:                bool
    fp_risk_flag:            str            = "none"
    fn_risk_flag:            str            = "none"
    channel:                 str            = "online"
    pos_terminal_id:         Optional[str]  = None
    store_id:                Optional[str]  = None
    cashier_id:              Optional[str]  = None
    card_present:            Optional[bool] = None
    cashier_override_flag:   Optional[bool] = None

    def to_dict(self) -> dict:
        return asdict(self)
