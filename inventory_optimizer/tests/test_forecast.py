"""需要予測モジュールのユニットテスト。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.generate_sample_data import generate_sales_data
from src.forecast import DemandForecaster, _build_features, FEATURE_COLS


@pytest.fixture(scope="module")
def sample_df() -> pd.DataFrame:
    return generate_sales_data(n_weeks=52, random_seed=42)


@pytest.fixture(scope="module")
def fitted_forecaster(sample_df: pd.DataFrame) -> DemandForecaster:
    fc = DemandForecaster()
    fc.fit(sample_df)
    return fc


class TestGenerateSalesData:
    def test_columns(self, sample_df: pd.DataFrame) -> None:
        assert set(sample_df.columns) >= {"date", "sku", "sales", "price", "is_holiday"}

    def test_skus(self, sample_df: pd.DataFrame) -> None:
        assert set(sample_df["sku"].unique()) == {"商品A", "商品B", "商品C", "商品D", "商品E"}

    def test_weeks_per_sku(self, sample_df: pd.DataFrame) -> None:
        for sku, group in sample_df.groupby("sku"):
            assert len(group) == 52, f"{sku} should have 52 weeks"

    def test_sales_non_negative(self, sample_df: pd.DataFrame) -> None:
        assert (sample_df["sales"] >= 0).all()

    def test_is_holiday_binary(self, sample_df: pd.DataFrame) -> None:
        assert set(sample_df["is_holiday"].unique()) <= {0, 1}


class TestBuildFeatures:
    def test_feature_columns_present(self, sample_df: pd.DataFrame) -> None:
        sku_df = sample_df[sample_df["sku"] == "商品A"].copy()
        featured = _build_features(sku_df)
        for col in FEATURE_COLS:
            assert col in featured.columns, f"Missing column: {col}"

    def test_lag_shift(self, sample_df: pd.DataFrame) -> None:
        sku_df = sample_df[sample_df["sku"] == "商品A"].copy()
        featured = _build_features(sku_df).dropna(subset=["lag_1"])
        # lag_1 should equal previous row's sales
        for i in range(1, len(featured)):
            assert featured["lag_1"].iloc[i] == featured["sales"].iloc[i - 1]

    def test_rolling_mean_4(self, sample_df: pd.DataFrame) -> None:
        sku_df = sample_df[sample_df["sku"] == "商品A"].copy()
        featured = _build_features(sku_df).dropna(subset=["rolling_mean_4"])
        assert featured["rolling_mean_4"].notna().all()


class TestDemandForecaster:
    def test_fit_creates_models_for_all_skus(
        self, fitted_forecaster: DemandForecaster, sample_df: pd.DataFrame
    ) -> None:
        skus = sample_df["sku"].unique().tolist()
        for sku in skus:
            assert sku in fitted_forecaster._models

    def test_predict_returns_series(self, fitted_forecaster: DemandForecaster) -> None:
        forecast = fitted_forecaster.predict("商品A", n_weeks=4)
        assert isinstance(forecast, pd.Series)
        assert len(forecast) == 4

    def test_predict_non_negative(self, fitted_forecaster: DemandForecaster) -> None:
        for sku in ["商品A", "商品B", "商品C"]:
            forecast = fitted_forecaster.predict(sku, n_weeks=4)
            assert (forecast >= 0).all(), f"{sku}: forecast should be non-negative"

    def test_predict_raises_for_unknown_sku(self, fitted_forecaster: DemandForecaster) -> None:
        with pytest.raises(ValueError, match="モデルが見つかりません"):
            fitted_forecaster.predict("存在しないSKU")

    def test_predict_n_weeks_length(self, fitted_forecaster: DemandForecaster) -> None:
        for n in [1, 4, 8]:
            forecast = fitted_forecaster.predict("商品A", n_weeks=n)
            assert len(forecast) == n

    def test_predict_dates_are_future(self, fitted_forecaster: DemandForecaster) -> None:
        history = fitted_forecaster.get_history("商品A")
        last_date = pd.to_datetime(history["date"]).max()
        forecast = fitted_forecaster.predict("商品A", n_weeks=4)
        assert (forecast.index > last_date).all()

    def test_predict_with_confidence_columns(self, fitted_forecaster: DemandForecaster) -> None:
        df = fitted_forecaster.predict_with_confidence("商品A", n_weeks=4)
        assert set(df.columns) == {"forecast", "lower", "upper"}
        assert len(df) == 4

    def test_predict_with_confidence_lower_le_upper(
        self, fitted_forecaster: DemandForecaster
    ) -> None:
        df = fitted_forecaster.predict_with_confidence("商品A")
        assert (df["lower"] <= df["upper"]).all()

    def test_get_feature_importance_shape(self, fitted_forecaster: DemandForecaster) -> None:
        df = fitted_forecaster.get_feature_importance()
        assert "feature" in df.columns
        assert "importance" in df.columns
        assert len(df) == len(FEATURE_COLS) * len(fitted_forecaster._models)
