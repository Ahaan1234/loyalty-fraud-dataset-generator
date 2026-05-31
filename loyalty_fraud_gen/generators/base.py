"""Abstract base generator with shared helpers."""
from __future__ import annotations

import math
from abc import ABC
from typing import Optional

import numpy as np
from faker import Faker

from loyalty_fraud_gen.config import Config


class BaseGenerator(ABC):
    """All generators share rng, config, and a Faker instance."""

    def __init__(self, rng: np.random.Generator, config: Config, fake: Faker) -> None:
        self.rng  = rng
        self.cfg  = config
        self.fake = fake

    # ── Random helpers ────────────────────────────────────────────────────────

    def weighted_choice(self, options: list, weights: list):
        """Weighted random choice using the generator's RNG."""
        n   = min(len(options), len(weights))
        w   = np.array(weights[:n], dtype=float)
        w  /= w.sum()
        idx = int(self.rng.choice(n, p=w))
        return options[idx]

    def random_ip(self, kind: str = "domestic") -> str:
        rng = self.rng
        if kind == "foreign":
            pfx = int(rng.choice([5, 31, 46, 77, 91, 185]))
            return f"{pfx}.{int(rng.integers(0,256))}.{int(rng.integers(0,256))}.{int(rng.integers(1,255))}"
        if kind == "vpn":
            pfx = int(rng.choice([104, 107, 108, 198]))
            return f"{pfx}.{int(rng.integers(0,256))}.{int(rng.integers(0,256))}.{int(rng.integers(1,255))}"
        if rng.random() < 0.3:
            return self.fake.ipv4_private()
        return self.fake.ipv4_public()

    def random_device(self, kind: str = "legit") -> str:
        if kind == "bot":
            return str(self.rng.choice([
                "Unknown Device", "HeadlessChrome/110",
                "Python-requests/2.28", "curl/7.88",
            ]))
        return str(self.rng.choice([
            "iPhone 14", "Samsung Galaxy S23", "Chrome/Windows", "Safari/macOS",
            "Chrome/Android", "Firefox/Linux", "iPad Pro", "Pixel 7",
        ]))

    def sample_purchase(
        self,
        baseline_spend: float,
        category_override: Optional[str] = None,
    ) -> tuple[str, float, int]:
        """Return (category, amount_usd, points_earned)."""
        cfg = self.cfg
        if category_override:
            cat = category_override
        else:
            cats    = list(cfg.product_categories.keys())
            weights = [cfg.product_categories[c][2] for c in cats]
            cat     = self.weighted_choice(cats, weights)
        lo, hi, _ = cfg.product_categories[cat]
        amount = round(min(hi, max(lo, self.rng.lognormal(
            mean=math.log(max(baseline_spend, lo + 1)), sigma=0.5
        ))), 2)
        return cat, amount, int(amount * cfg.points_per_dollar)
