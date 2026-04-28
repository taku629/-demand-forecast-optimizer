"""pytest 共通フィクスチャ。"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.generate_sample_data import generate_sales_data
from src.forecast import DemandForecaster


@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    """52週分のサンプル販売データ（セッションスコープ）。"""
    return generate_sales_data(n_weeks=52, random_seed=42)


@pytest.fixture(scope="session")
def fitted_forecaster(sample_df: pd.DataFrame) -> DemandForecaster:
    """学習済み DemandForecaster（セッションスコープ）。"""
    fc = DemandForecaster()
    fc.fit(sample_df)
    return fc


@pytest.fixture
def simple_demand() -> pd.Series:
    """4週間のシンプルな需要予測Series。"""
    dates = pd.date_range("2025-01-05", periods=4, freq="W-SUN")
    return pd.Series([100, 120, 90, 110], index=dates)


@pytest.fixture
def default_optimizer_params() -> dict:
    """最適化デフォルトパラメータ。"""
    return {
        "holding_cost": 10.0,
        "order_fixed_cost": 10_000.0,
        "stockout_penalty": 500.0,
        "warehouse_capacity": 500,
        "min_order_lot": 20,
    }
