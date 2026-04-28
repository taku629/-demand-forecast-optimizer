"""在庫最適化モジュール。

PuLPによる整数線形計画法で最適発注スケジュールを算出する。
"""

from __future__ import annotations

import pulp
import pandas as pd

from src.types import CostBreakdown, OptimizationResult


class InventoryOptimizer:
    """PuLPで整数線形計画を解く在庫最適化クラス。

    目的関数（最小化）：
        発注コスト（発注固定費 × 発注回数）
        + 保管コスト（1個あたり × 在庫量 × 週数）
        + 欠品ペナルティ（1個あたり × 不足量）

    Example:
        >>> optimizer = InventoryOptimizer()
        >>> result = optimizer.optimize(
        ...     demand_forecast=forecast_series,
        ...     current_stock=100,
        ...     holding_cost=5.0,
        ...     order_fixed_cost=10000.0,
        ...     stockout_penalty=500.0,
        ...     warehouse_capacity=300,
        ...     min_order_lot=10,
        ... )
    """

    def optimize(
        self,
        demand_forecast: pd.Series,
        current_stock: int,
        holding_cost: float,
        order_fixed_cost: float,
        stockout_penalty: float,
        warehouse_capacity: int,
        min_order_lot: int,
    ) -> OptimizationResult:
        """整数線形計画で最適発注スケジュールを算出する。

        Args:
            demand_forecast: 週次需要予測値のSeries（indexは日付）。
            current_stock: 現在の在庫数（個）。
            holding_cost: 1個あたり週次保管コスト（円）。
            order_fixed_cost: 1回の発注固定費（円）。
            stockout_penalty: 1個あたり欠品ペナルティ（円）。
            warehouse_capacity: 倉庫最大容量（個）。
            min_order_lot: 最小発注ロット（個）。

        Returns:
            OptimizationResult TypedDict。
        """
        demands = [max(0, round(d)) for d in demand_forecast.values]
        n_weeks = len(demands)
        dates = demand_forecast.index.tolist()
        big_m = warehouse_capacity

        prob = pulp.LpProblem("inventory_optimization", pulp.LpMinimize)

        order_qty = [
            pulp.LpVariable(f"order_qty_{t}", lowBound=0, cat="Integer")
            for t in range(n_weeks)
        ]
        stock = [
            pulp.LpVariable(f"stock_{t}", lowBound=0, upBound=warehouse_capacity, cat="Integer")
            for t in range(n_weeks)
        ]
        shortage = [
            pulp.LpVariable(f"shortage_{t}", lowBound=0, cat="Integer")
            for t in range(n_weeks)
        ]
        order_flag = [
            pulp.LpVariable(f"order_flag_{t}", cat="Binary")
            for t in range(n_weeks)
        ]

        # 目的関数
        prob += (
            pulp.lpSum(holding_cost * stock[t] for t in range(n_weeks))
            + pulp.lpSum(order_fixed_cost * order_flag[t] for t in range(n_weeks))
            + pulp.lpSum(stockout_penalty * shortage[t] for t in range(n_weeks))
        )

        for t in range(n_weeks):
            prev_stock = current_stock if t == 0 else stock[t - 1]
            prob += stock[t] == prev_stock + order_qty[t] - demands[t] + shortage[t]
            prob += order_qty[t] <= big_m * order_flag[t]
            prob += order_qty[t] >= min_order_lot * order_flag[t]
            prob += stock[t] <= warehouse_capacity

        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        status = pulp.LpStatus[prob.status]

        if prob.status != 1:
            return self._fallback_schedule(
                demands, current_stock, holding_cost, order_fixed_cost,
                stockout_penalty, warehouse_capacity, min_order_lot, dates, status,
            )

        order_schedule = [max(0, round(pulp.value(order_qty[t]) or 0)) for t in range(n_weeks)]
        stock_levels = [max(0, round(pulp.value(stock[t]) or 0)) for t in range(n_weeks)]
        shortage_vals = [max(0, round(pulp.value(shortage[t]) or 0)) for t in range(n_weeks)]

        holding_val = sum(holding_cost * s for s in stock_levels)
        ordering_val = sum(order_fixed_cost for o in order_schedule if o > 0)
        stockout_val = sum(stockout_penalty * s for s in shortage_vals)

        return OptimizationResult(
            order_schedule=order_schedule,
            stock_levels=stock_levels,
            shortage_levels=shortage_vals,
            total_cost=holding_val + ordering_val + stockout_val,
            cost_breakdown=CostBreakdown(
                holding=holding_val,
                ordering=ordering_val,
                stockout=stockout_val,
            ),
            status=status,
            dates=dates,
        )

    def _fallback_schedule(
        self,
        demands: list[int],
        current_stock: int,
        holding_cost: float,
        order_fixed_cost: float,
        stockout_penalty: float,
        warehouse_capacity: int,
        min_order_lot: int,
        dates: list,
        status: str,
    ) -> OptimizationResult:
        """最適化失敗時の安全在庫ベースの発注スケジュール。"""
        stock_sim = current_stock
        order_schedule: list[int] = []
        stock_levels: list[int] = []
        shortage_levels: list[int] = []

        for demand in demands:
            projected = stock_sim - demand
            if projected < 0:
                needed = -projected + min_order_lot
                order = (needed // min_order_lot + 1) * min_order_lot
                order = min(order, warehouse_capacity - stock_sim)
            else:
                order = 0
            order_schedule.append(order)
            after_order = min(stock_sim + order, warehouse_capacity)
            after_demand = after_order - demand
            shortage = max(0, -after_demand)
            stock_levels.append(max(0, after_demand))
            shortage_levels.append(shortage)
            stock_sim = max(0, after_demand)

        holding_val = sum(holding_cost * s for s in stock_levels)
        ordering_val = sum(order_fixed_cost for o in order_schedule if o > 0)
        stockout_val = sum(stockout_penalty * s for s in shortage_levels)

        return OptimizationResult(
            order_schedule=order_schedule,
            stock_levels=stock_levels,
            shortage_levels=shortage_levels,
            total_cost=holding_val + ordering_val + stockout_val,
            cost_breakdown=CostBreakdown(
                holding=holding_val,
                ordering=ordering_val,
                stockout=stockout_val,
            ),
            status=f"Fallback ({status})",
            dates=dates,
        )
