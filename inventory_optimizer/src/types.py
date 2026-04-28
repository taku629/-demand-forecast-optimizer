"""プロジェクト共通の型定義。"""

from __future__ import annotations

from typing import TypedDict


class CostBreakdown(TypedDict):
    """コスト内訳の型。"""

    holding: float
    ordering: float
    stockout: float


class OptimizationResult(TypedDict):
    """InventoryOptimizer.optimize() の戻り値型。"""

    order_schedule: list[int]
    stock_levels: list[int]
    shortage_levels: list[int]
    total_cost: float
    cost_breakdown: CostBreakdown
    status: str
    dates: list[object]


class CVScore(TypedDict):
    """交差検証スコアの型。"""

    sku: str
    rmse_mean: float
    rmse_std: float
    mae_mean: float
    n_splits: int
