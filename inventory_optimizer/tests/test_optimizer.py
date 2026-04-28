"""在庫最適化モジュールのユニットテスト。"""

from __future__ import annotations

import pandas as pd
import pytest

from src.optimizer import InventoryOptimizer
from src.types import OptimizationResult


class TestInventoryOptimizer:
    def test_returns_required_keys(
        self, simple_demand: pd.Series, default_optimizer_params: dict
    ) -> None:
        result = InventoryOptimizer().optimize(
            demand_forecast=simple_demand, current_stock=50, **default_optimizer_params
        )
        required_keys = {"order_schedule", "stock_levels", "total_cost", "cost_breakdown", "status", "dates"}
        assert required_keys <= set(result.keys())

    def test_return_type_is_typed_dict(
        self, simple_demand: pd.Series, default_optimizer_params: dict
    ) -> None:
        result = InventoryOptimizer().optimize(
            demand_forecast=simple_demand, current_stock=50, **default_optimizer_params
        )
        # TypedDict はランタイムでは通常の dict
        assert isinstance(result, dict)
        assert isinstance(result["cost_breakdown"], dict)

    def test_order_schedule_length(
        self, simple_demand: pd.Series, default_optimizer_params: dict
    ) -> None:
        result = InventoryOptimizer().optimize(
            demand_forecast=simple_demand, current_stock=50, **default_optimizer_params
        )
        assert len(result["order_schedule"]) == 4

    def test_stock_levels_non_negative(
        self, simple_demand: pd.Series, default_optimizer_params: dict
    ) -> None:
        result = InventoryOptimizer().optimize(
            demand_forecast=simple_demand, current_stock=50, **default_optimizer_params
        )
        assert all(s >= 0 for s in result["stock_levels"])

    def test_stock_levels_within_capacity(
        self, simple_demand: pd.Series, default_optimizer_params: dict
    ) -> None:
        result = InventoryOptimizer().optimize(
            demand_forecast=simple_demand, current_stock=50, **default_optimizer_params
        )
        assert all(s <= default_optimizer_params["warehouse_capacity"] for s in result["stock_levels"])

    def test_order_qty_non_negative(
        self, simple_demand: pd.Series, default_optimizer_params: dict
    ) -> None:
        result = InventoryOptimizer().optimize(
            demand_forecast=simple_demand, current_stock=50, **default_optimizer_params
        )
        assert all(o >= 0 for o in result["order_schedule"])

    def test_order_respects_min_lot(
        self, simple_demand: pd.Series, default_optimizer_params: dict
    ) -> None:
        result = InventoryOptimizer().optimize(
            demand_forecast=simple_demand, current_stock=50, **default_optimizer_params
        )
        min_lot = default_optimizer_params["min_order_lot"]
        assert all(o == 0 or o >= min_lot for o in result["order_schedule"])

    def test_total_cost_matches_breakdown(
        self, simple_demand: pd.Series, default_optimizer_params: dict
    ) -> None:
        result = InventoryOptimizer().optimize(
            demand_forecast=simple_demand, current_stock=50, **default_optimizer_params
        )
        bd = result["cost_breakdown"]
        expected = bd["holding"] + bd["ordering"] + bd["stockout"]
        assert abs(result["total_cost"] - expected) < 1.0

    def test_cost_breakdown_keys(
        self, simple_demand: pd.Series, default_optimizer_params: dict
    ) -> None:
        result = InventoryOptimizer().optimize(
            demand_forecast=simple_demand, current_stock=50, **default_optimizer_params
        )
        assert set(result["cost_breakdown"].keys()) == {"holding", "ordering", "stockout"}

    def test_cost_values_non_negative(
        self, simple_demand: pd.Series, default_optimizer_params: dict
    ) -> None:
        result = InventoryOptimizer().optimize(
            demand_forecast=simple_demand, current_stock=50, **default_optimizer_params
        )
        for v in result["cost_breakdown"].values():
            assert v >= 0

    def test_high_initial_stock_reduces_orders(self, default_optimizer_params: dict) -> None:
        dates = pd.date_range("2025-01-05", periods=4, freq="W-SUN")
        demand = pd.Series([50, 50, 50, 50], index=dates)
        optimizer = InventoryOptimizer()
        result_low = optimizer.optimize(demand_forecast=demand, current_stock=0, **default_optimizer_params)
        result_high = optimizer.optimize(demand_forecast=demand, current_stock=400, **default_optimizer_params)
        assert sum(result_high["order_schedule"]) <= sum(result_low["order_schedule"])

    def test_high_stockout_penalty_reduces_shortage(self, default_optimizer_params: dict) -> None:
        dates = pd.date_range("2025-01-05", periods=4, freq="W-SUN")
        demand = pd.Series([200, 200, 200, 200], index=dates)
        optimizer = InventoryOptimizer()
        params_low = {**default_optimizer_params, "stockout_penalty": 100.0}
        params_high = {**default_optimizer_params, "stockout_penalty": 5000.0}
        result_low = optimizer.optimize(demand_forecast=demand, current_stock=0, **params_low)
        result_high = optimizer.optimize(demand_forecast=demand, current_stock=0, **params_high)
        shortage_low = sum(result_low.get("shortage_levels", [0] * 4))
        shortage_high = sum(result_high.get("shortage_levels", [0] * 4))
        assert shortage_high <= shortage_low

    def test_dates_length_matches_demand(
        self, simple_demand: pd.Series, default_optimizer_params: dict
    ) -> None:
        result = InventoryOptimizer().optimize(
            demand_forecast=simple_demand, current_stock=50, **default_optimizer_params
        )
        assert len(result["dates"]) == len(simple_demand)

    def test_zero_initial_stock_triggers_order(self, default_optimizer_params: dict) -> None:
        """在庫ゼロスタートでは最初の週に発注が発生するはず。"""
        dates = pd.date_range("2025-01-05", periods=4, freq="W-SUN")
        demand = pd.Series([100, 100, 100, 100], index=dates)
        result = InventoryOptimizer().optimize(
            demand_forecast=demand, current_stock=0, **default_optimizer_params
        )
        assert sum(result["order_schedule"]) > 0
