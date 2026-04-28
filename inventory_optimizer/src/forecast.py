"""需要予測モジュール。

LightGBMを使ったSKUごとの週次需要予測を提供する。
"""

from __future__ import annotations

import warnings
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore", category=UserWarning)


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """時系列特徴量を構築する。

    Args:
        df: date, sku, sales, price, is_holiday カラムを持つDataFrame。

    Returns:
        特徴量カラムを追加したDataFrame。
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month

    # ラグ特徴量
    df["lag_1"] = df["sales"].shift(1)
    df["lag_4"] = df["sales"].shift(4)
    df["lag_8"] = df["sales"].shift(8)

    # 移動平均
    df["rolling_mean_4"] = df["sales"].shift(1).rolling(4).mean()

    return df


FEATURE_COLS = [
    "week_of_year",
    "month",
    "is_holiday",
    "lag_1",
    "lag_4",
    "lag_8",
    "rolling_mean_4",
]


class DemandForecaster:
    """SKUごとにLightGBMモデルを学習し需要予測を行うクラス。

    Example:
        >>> forecaster = DemandForecaster()
        >>> forecaster.fit(df)
        >>> forecast = forecaster.predict("商品A", n_weeks=4)
    """

    def __init__(self, lgb_params: dict[str, Any] | None = None) -> None:
        """初期化。

        Args:
            lgb_params: LightGBMのハイパーパラメータ。Noneの場合はデフォルト値を使用。
        """
        self._params: dict[str, Any] = lgb_params or {
            "objective": "regression",
            "metric": "rmse",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "n_estimators": 200,
        }
        self._models: dict[str, lgb.LGBMRegressor] = {}
        self._history: dict[str, pd.DataFrame] = {}

    def fit(self, df: pd.DataFrame) -> None:
        """SKUごとにLightGBMモデルをfitする。

        Args:
            df: date, sku, sales, price, is_holiday カラムを持つDataFrame。
        """
        for sku, group in df.groupby("sku"):
            featured = _build_features(group)
            featured = featured.dropna(subset=FEATURE_COLS)
            X = featured[FEATURE_COLS]
            y = featured["sales"]

            model = lgb.LGBMRegressor(**self._params)
            model.fit(X, y)
            self._models[sku] = model
            self._history[sku] = _build_features(group)  # 生の履歴も保持

    def predict(self, sku: str, n_weeks: int = 4) -> pd.Series:
        """指定SKUの今後n_weeks週の需要予測値を返す。

        Args:
            sku: 対象SKU名。
            n_weeks: 予測する週数。

        Returns:
            予測値のSeries（indexは予測日付）。

        Raises:
            ValueError: 指定SKUのモデルが存在しない場合。
        """
        if sku not in self._models:
            raise ValueError(f"SKU '{sku}' のモデルが見つかりません。先にfit()を実行してください。")

        model = self._models[sku]
        history = self._history[sku].copy()
        history["date"] = pd.to_datetime(history["date"])
        history = history.sort_values("date").reset_index(drop=True)

        last_date = history["date"].iloc[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(weeks=1), periods=n_weeks, freq="W-SUN"
        )

        # is_holidayを推定（正月・GW・お盆・年末）
        def _is_holiday(date: pd.Timestamp) -> int:
            week = date.isocalendar().week
            month = date.month
            if month == 1 and week <= 2:
                return 1
            if month in (4, 5) and 17 <= week <= 19:
                return 1
            if month == 8 and 32 <= week <= 34:
                return 1
            if month == 12 and week >= 51:
                return 1
            return 0

        predictions: list[float] = []
        sales_buffer = history["sales"].tolist()

        for date in future_dates:
            lag_1 = sales_buffer[-1] if len(sales_buffer) >= 1 else 0
            lag_4 = sales_buffer[-4] if len(sales_buffer) >= 4 else np.mean(sales_buffer)
            lag_8 = sales_buffer[-8] if len(sales_buffer) >= 8 else np.mean(sales_buffer)
            rolling_mean_4 = np.mean(sales_buffer[-4:]) if len(sales_buffer) >= 4 else np.mean(sales_buffer)

            row = pd.DataFrame(
                [
                    {
                        "week_of_year": date.isocalendar().week,
                        "month": date.month,
                        "is_holiday": _is_holiday(date),
                        "lag_1": lag_1,
                        "lag_4": lag_4,
                        "lag_8": lag_8,
                        "rolling_mean_4": rolling_mean_4,
                    }
                ]
            )
            pred = float(model.predict(row)[0])
            pred = max(0.0, pred)
            predictions.append(pred)
            sales_buffer.append(pred)

        return pd.Series(predictions, index=future_dates, name=f"{sku}_forecast")

    def predict_with_confidence(
        self, sku: str, n_weeks: int = 4, ci_multiplier: float = 1.96
    ) -> pd.DataFrame:
        """需要予測値と信頼区間を返す。

        Args:
            sku: 対象SKU名。
            n_weeks: 予測する週数。
            ci_multiplier: 信頼区間の係数（デフォルト1.96で95%区間）。

        Returns:
            forecast, lower, upper カラムを持つDataFrame。
        """
        forecast = self.predict(sku, n_weeks)

        # 訓練残差からRMSEを計算して信頼区間を推定
        history = self._history[sku].copy()
        featured = _build_features(history).dropna(subset=FEATURE_COLS)
        X = featured[FEATURE_COLS]
        y_true = featured["sales"].values
        model = self._models[sku]
        y_pred = model.predict(X)
        residuals = y_true - y_pred
        rmse = float(np.sqrt(np.mean(residuals**2)))

        lower = (forecast - ci_multiplier * rmse).clip(lower=0)
        upper = forecast + ci_multiplier * rmse

        return pd.DataFrame(
            {"forecast": forecast, "lower": lower, "upper": upper},
            index=forecast.index,
        )

    def get_feature_importance(self) -> pd.DataFrame:
        """全SKUの特徴量重要度をDataFrameで返す。

        Returns:
            feature, sku, importance カラムを持つDataFrame。
        """
        records = []
        for sku, model in self._models.items():
            importances = model.feature_importances_
            for feat, imp in zip(FEATURE_COLS, importances):
                records.append({"feature": feat, "sku": sku, "importance": imp})
        return pd.DataFrame(records)

    def get_history(self, sku: str) -> pd.DataFrame:
        """指定SKUの学習に使用した履歴データを返す。

        Args:
            sku: 対象SKU名。

        Returns:
            履歴DataFrame。
        """
        if sku not in self._history:
            raise ValueError(f"SKU '{sku}' の履歴データが見つかりません。")
        return self._history[sku].copy()
