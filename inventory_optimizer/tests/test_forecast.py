"""需要予測モジュールのユニットテスト。"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.forecast import FEATURE_COLS, DemandForecaster, _build_features


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
        featured = _build_features(sku_df).dropna(subset=["lag_1"]).reset_index(drop=True)
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
        for sku in sample_df["sku"].unique():
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
            assert len(fitted_forecaster.predict("商品A", n_weeks=n)) == n

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

    def test_predict_with_confidence_lower_non_negative(
        self, fitted_forecaster: DemandForecaster
    ) -> None:
        df = fitted_forecaster.predict_with_confidence("商品A")
        assert (df["lower"] >= 0).all()

    def test_get_feature_importance_shape(self, fitted_forecaster: DemandForecaster) -> None:
        df = fitted_forecaster.get_feature_importance()
        assert "feature" in df.columns
        assert "importance" in df.columns
        assert len(df) == len(FEATURE_COLS) * len(fitted_forecaster._models)

    def test_cross_validate_returns_cv_score(self, fitted_forecaster: DemandForecaster) -> None:
        score = fitted_forecaster.cross_validate("商品A", n_splits=3)
        assert score["sku"] == "商品A"
        assert score["n_splits"] == 3
        assert score["rmse_mean"] > 0
        assert score["rmse_std"] >= 0
        assert score["mae_mean"] > 0

    def test_cross_validate_rmse_reasonable(self, fitted_forecaster: DemandForecaster) -> None:
        """RMSE が平均販売数の50%以下であること（簡易品質チェック）。"""
        history = fitted_forecaster.get_history("商品A")
        mean_sales = history["sales"].mean()
        score = fitted_forecaster.cross_validate("商品A", n_splits=3)
        assert score["rmse_mean"] < mean_sales * 0.5

    def test_save_and_load(
        self, fitted_forecaster: DemandForecaster, sample_df: pd.DataFrame
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "forecaster.pkl"
            fitted_forecaster.save(path)
            loaded = DemandForecaster.load(path)

        # 同じ予測値が返ること
        original = fitted_forecaster.predict("商品A", n_weeks=4)
        restored = loaded.predict("商品A", n_weeks=4)
        assert np.allclose(original.values, restored.values, rtol=1e-5)
