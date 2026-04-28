"""utils モジュールのユニットテスト。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.utils import build_recommendation_table, format_currency, load_sales_data


class TestFormatCurrency:
    def test_basic(self) -> None:
        assert format_currency(12345.0) == "¥12,345"

    def test_zero(self) -> None:
        assert format_currency(0) == "¥0"

    def test_large_number(self) -> None:
        assert format_currency(1_000_000) == "¥1,000,000"

    def test_decimal_truncated(self) -> None:
        assert format_currency(99.9) == "¥100"


class TestLoadSalesData:
    def test_load_existing_csv(self) -> None:
        data_path = Path(__file__).parent.parent / "data" / "sales_history.csv"
        if not data_path.exists():
            pytest.skip("sales_history.csv が未生成")
        df = load_sales_data(data_path)
        assert set(df.columns) >= {"date", "sku", "sales", "price", "is_holiday"}
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_raises_for_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_sales_data("/nonexistent/path/data.csv")

    def test_default_path_raises_if_missing(self, tmp_path: Path) -> None:
        """存在しないデフォルトパスで FileNotFoundError が出ること。"""
        import src.utils as utils_mod
        original = utils_mod.load_sales_data

        def patched(csv_path: object = None) -> pd.DataFrame:
            if csv_path is None:
                raise FileNotFoundError("no default")
            return original(csv_path)  # type: ignore[arg-type]

        with pytest.raises(FileNotFoundError):
            patched()


class TestBuildRecommendationTable:
    def test_column_names(self) -> None:
        dates = pd.date_range("2025-01-05", periods=4, freq="W-SUN")
        table = build_recommendation_table(
            dates=dates.tolist(),
            order_schedule=[50, 0, 30, 0],
            demand_forecast=[100.0, 80.0, 90.0, 70.0],
            stock_levels=[50, 20, 40, 10],
            holding_cost=10.0,
            order_fixed_cost=5000.0,
        )
        expected_cols = {"週", "発注量 (個)", "予測需要 (個)", "予想在庫 (個)", "週次コスト (円)"}
        assert set(table.columns) == expected_cols

    def test_row_count(self) -> None:
        dates = pd.date_range("2025-01-05", periods=4, freq="W-SUN")
        table = build_recommendation_table(
            dates=dates.tolist(),
            order_schedule=[50, 0, 30, 0],
            demand_forecast=[100.0, 80.0, 90.0, 70.0],
            stock_levels=[50, 20, 40, 10],
            holding_cost=10.0,
            order_fixed_cost=5000.0,
        )
        assert len(table) == 4

    def test_order_week_includes_fixed_cost(self) -> None:
        """発注がある週はコストに固定費が含まれること。"""
        dates = pd.date_range("2025-01-05", periods=2, freq="W-SUN")
        table = build_recommendation_table(
            dates=dates.tolist(),
            order_schedule=[100, 0],
            demand_forecast=[100.0, 100.0],
            stock_levels=[100, 50],
            holding_cost=10.0,
            order_fixed_cost=5000.0,
        )
        # 1週目: 保管10×100 + 固定5000 = ¥6,000
        assert "6,000" in table.iloc[0]["週次コスト (円)"]
        # 2週目: 保管10×50 + 固定0 = ¥500
        assert "500" in table.iloc[1]["週次コスト (円)"]

    def test_date_format(self) -> None:
        dates = pd.date_range("2025-01-05", periods=1, freq="W-SUN")
        table = build_recommendation_table(
            dates=dates.tolist(),
            order_schedule=[0],
            demand_forecast=[50.0],
            stock_levels=[100],
            holding_cost=5.0,
            order_fixed_cost=0.0,
        )
        assert table.iloc[0]["週"] == "2025-01-05"
