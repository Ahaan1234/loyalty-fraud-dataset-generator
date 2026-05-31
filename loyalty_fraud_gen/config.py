"""All constants and the Config dataclass."""
from __future__ import annotations

from dataclasses import dataclass, field

# ── Retailers ──────────────────────────────────────────────────────────────────
RETAILERS = ["ShopMart", "FreshGrocer", "TechZone", "FashionHub", "HomeNest"]

# ── Product categories: (min_spend, max_spend, sample_weight) ─────────────────
PRODUCT_CATEGORIES: dict[str, tuple[int, int, float]] = {
    "groceries":   (5,   120,  0.30),
    "electronics": (50,  2000, 0.10),
    "clothing":    (20,  300,  0.20),
    "homewares":   (15,  500,  0.15),
    "beauty":      (10,  150,  0.15),
    "sports":      (25,  400,  0.10),
}

POINTS_PER_DOLLAR = 10      # base earn rate
REDEMPTION_RATE   = 0.01    # 1 point = $0.01

# ── Hour-of-day probability weights (one entry per hour 0–23) ─────────────────
# NOTE: ATO_HOUR_W intentionally has 25 entries (matches original); the 25th
# is silently dropped when zipped against range(24) — preserved as-is.
LEGIT_HOUR_W = (
    [0.003] * 5 + [0.01, 0.02, 0.05]
    + [0.07, 0.08, 0.08] + [0.07, 0.07, 0.06]
    + [0.06, 0.07, 0.07] + [0.07, 0.06, 0.05]
    + [0.04, 0.03, 0.02] + [0.01]
)
ATO_HOUR_W = (
    [0.06] * 6 + [0.05, 0.04, 0.04]
    + [0.04, 0.04, 0.04] + [0.04, 0.04, 0.04]
    + [0.04, 0.04, 0.04] + [0.05, 0.05, 0.05]
    + [0.05, 0.06, 0.06] + [0.05]
)
# Off-peak: heavy weight on early morning (0-5) and late evening (21-23)
OFFPEAK_HOUR_W = (
    [0.08] * 6          # 0-5  early morning
    + [0.01] * 6        # 6-11 morning
    + [0.01] * 6        # 12-17 afternoon
    + [0.01] * 3        # 18-20 evening
    + [0.07, 0.08, 0.09]  # 21-23 late evening
)

# ── Dirtiness parameters ───────────────────────────────────────────────────────
# (null_rate, duplicate_rate, jitter_sec, label_noise_rate, pts_drift_rate)
DIRT_PARAMS: dict[str, tuple] = {
    "low":    (0.02, 0.005, 30,   0.01, 0.05),
    "medium": (0.06, 0.015, 120,  0.03, 0.12),
    "high":   (0.12, 0.040, 600,  0.07, 0.25),
}

# ── In-store pool sizes ────────────────────────────────────────────────────────
N_STORES                = 20
MIN_CASHIERS_PER_STORE  = 5
MAX_CASHIERS_PER_STORE  = 12
MIN_TERMINALS_PER_STORE = 2
MAX_TERMINALS_PER_STORE = 6


@dataclass
class Config:
    """Runtime-overridable configuration; defaults mirror the module constants."""
    retailers:               list  = field(default_factory=lambda: list(RETAILERS))
    product_categories:      dict  = field(default_factory=lambda: dict(PRODUCT_CATEGORIES))
    points_per_dollar:       int   = POINTS_PER_DOLLAR
    redemption_rate:         float = REDEMPTION_RATE
    legit_hour_w:            list  = field(default_factory=lambda: list(LEGIT_HOUR_W))
    ato_hour_w:              list  = field(default_factory=lambda: list(ATO_HOUR_W))
    offpeak_hour_w:          list  = field(default_factory=lambda: list(OFFPEAK_HOUR_W))
    dirt_params:             dict  = field(default_factory=lambda: dict(DIRT_PARAMS))
    n_stores:                int   = N_STORES
    min_cashiers_per_store:  int   = MIN_CASHIERS_PER_STORE
    max_cashiers_per_store:  int   = MAX_CASHIERS_PER_STORE
    min_terminals_per_store: int   = MIN_TERMINALS_PER_STORE
    max_terminals_per_store: int   = MAX_TERMINALS_PER_STORE
