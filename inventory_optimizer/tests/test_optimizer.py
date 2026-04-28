"""在庫最適化モジュールのユニットテスト。"""

from __future__ import annotations

import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.optimizer import InventoryOptimizer


@pytest.fixture
def default_params() -> dict:
    return {
        "holding_cost": 10.0,
        "order_fixed_cost": 10_000.0,
        "stockout_penalty": 500.0,
        "warehouse_capacity": 500,
        "min_order_lot": 20,
    }


@pytest.fixture
def simple_demand() -> pd.Series:
    dates = pd.date_range("2025-01-05", periods=4, freq="W-SUN")
    return pd.Series([100, 120, 90, 110], index=dates)


class TestInventoryOptimizer:
    def test_returns_required_keys(
        self, simple_demand: pd.Series, default_params: dict
    ) -> None:
        optimizer = InventoryOptimizer()
        result = optimizer.optimize(
            demand_forecast=simple_demand,
            current_stock=50,
            **default_params,
        )
        required_keys = {"order_schedule", "stock_levels", "total_cost", "cost_breakdown", "status", "dates"}
        assert required_keys <= set(result.keys())

    def test_order_schedule_length(
        self, simple_demand: pd.Series, default_params: dict
    ) -> None:
        optimizer = InventoryOptimizer()
        result = optimizer.optimize(
            demand_forecast=simple_demand,
            current_stock=50,
            **default_params,
        )
        assert len(result["order_schedule"]) == 4

    def test_stock_levels_non_negative(
        self, simple_demand: pd.Series, default_params: dict
    ) -> None:
        optimizer = InventoryOptimizer()
        result = optimizer.optimize(
            demand_forecast=simple_demand,
            current_stock=50,
            **default_params,
        )
        assert all(s >= 0 for s in result["stock_levels"])

    def test_stock_levels_within_capacity(
        self, simple_demand: pd.Series, default_params: dict
    ) -> None:
        optimizer = InventoryOptimizer()
        result = optimizer.optimize(
            demand_forecast=simple_demand,
            current_stock=50,
            **default_params,
        )
        capacity = default_params["warehouse_capacity"]
        assert all(s <= capacity for s in result["stock_levels"])

    def test_order_qty_non_negative(
        self, simple_demand: pd.Series, default_params: dict
    ) -> None:
        optimizer = InventoryOptimizer()
        result = optimizer.optimize(
            demand_forecast=simple_demand,
            current_stock=50,
            **default_params,
        )
        assert all(o >= 0 for o in result["order_schedule"])

    def test_order_respects_min_lot(
        self, simple_demand: pd.Series, default_params: dict
    ) -> None:
        optimizer = InventoryOptimizer()
        result = optimizer.optimize(
            demand_forecast=simple_demand,
            current_stock=50,
            **default_params,
        )
        min_lot = default_params["min_order_lot"]
        for order in result["order_schedule"]:
            assert order == 0 or order >= min_lot

    def test_total_cost_matches_breakdown(
        self, simple_demand: pd.Series, default_params: dict
    ) -> None:
        optimizer = InventoryOptimizer()
        result = optimizer.optimize(
            demand_forecast=simple_demand,
            current_stock=50,
            **default_params,
        )
        breakdown = result["cost_breakdown"]
        expected_total = breakdown["holding"] + breakdown["ordering"] + breakdown["stockout"]
        assert abs(result["total_cost"] - expected_total) < 1.0  # 1円以内

    def test_cost_breakdown_keys(
        self, simple_demand: pd.Series, default_params: dict
    ) -> None:
        optimizer = InventoryOptimizer()
        result = optimizer.optimize(
            demand_forecast=simple_demand,
            current_stock=50,
            **default_params,
        )
        assert set(result["cost_breakdown"].keys()) == {"holding", "ordering", "stockout"}

    def test_high_initial_stock_reduces_orders(self, default_params: dict) -> None:
        """高在庫スタートでは発注が抑制される。"""
        dates = pd.date_range("2025-01-05", periods=4, freq="W-SUN")
        demand = pd.Series([50, 50, 50, 50], index=dates)
        optimizer = InventoryOptimizer()

        result_low = optimizer.optimize(demand_forecast=demand, current_stock=0, **default_params)
        result_high = optimizer.optimize(demand_forecast=demand, current_stock=400, **default_params)

        assert sum(result_high["order_schedule"]) <= sum(result_low["order_schedule"])

    def test_high_stockout_penalty_reduces_shortage(self, default_params: dict) -> None:
        """欠品ペナルティが高いほど欠品が減る傾向。"""
        dates = pd.date_range("2025-01-05", periods=4, freq="W-SUN")
        demand = pd.Series([200, 200, 200, 200], index=dates)
        optimizer = InventoryOptimizer()

        params_low = {**default_params, "stockout_penalty": 100.0}
        params_high = {**default_params, "stockout_penalty": 5000.0}

        result_low = optimizer.optimize(demand_forecast=demand, current_stock=0, **params_low)
        result_high = optimizer.optimize(demand_forecast=demand, current_stock=0, **params_high)

        shortage_low = sum(result_low.get("shortage_levels", [0] * 4))
        shortage_high = sum(result_high.get("shortage_levels", [0] * 4))
        assert shortage_high <= shortage_low

    def test_dates_length_matches_demand(
        self, simple_demand: pd.Series, default_params: dict
    ) -> None:
        optimizer = InventoryOptimizer()
        result = optimizer.optimize(
            demand_forecast=simple_demand,
            current_stock=50,
            **default_params,
        )
        assert len(result["dates"]) == len(simple_demand)
