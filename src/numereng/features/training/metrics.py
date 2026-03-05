"""Compatibility module alias for training scoring metrics helpers.

Prefer importing from `numereng.features.training.scoring.metrics`.
"""

from __future__ import annotations

import sys

from numereng.features.training.scoring import metrics as _scoring_metrics

sys.modules[__name__] = _scoring_metrics
