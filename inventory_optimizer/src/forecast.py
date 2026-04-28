"""需要予測モジュール。

LightGBMを使ったSKUごとの週次需要予測を提供する。
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from src.types import CVScore

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


FEATURE_COLS: list[str] = [
    "week_of_year",
    "month",
    "is_holiday",
    "lag_1",
    "lag_4",
    "lag_8",
    "rolling_mean_4",
]


def _estimate_holiday(date: pd.Timestamp) -> int:
    """日付から祝日フラグを推定する（正月・GW・お盆・年末）。"""
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


class DemandForecaster:
    """SKUごとにLightGBMモデルを学習し需要予測を行うクラス。

    Example:
        >>> forecaster = DemandForecaster()
        >>> forecaster.fit(df)
        >>> forecast = forecaster.predict("商品A", n_weeks=4)
        >>> scores = forecaster.cross_validate("商品A")
        >>> forecaster.save("models/forecaster.pkl")
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
            self._models[str(sku)] = model
            self._history[str(sku)] = _build_features(group)

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

        predictions: list[float] = []
        sales_buffer = history["sales"].tolist()

        for date in future_dates:
            n = len(sales_buffer)
            lag_1 = sales_buffer[-1] if n >= 1 else 0.0
            lag_4 = sales_buffer[-4] if n >= 4 else float(np.mean(sales_buffer))
            lag_8 = sales_buffer[-8] if n >= 8 else float(np.mean(sales_buffer))
            rolling_mean_4 = float(np.mean(sales_buffer[-4:])) if n >= 4 else float(np.mean(sales_buffer))

            row = pd.DataFrame(
                [
                    {
                        "week_of_year": date.isocalendar().week,
                        "month": date.month,
                        "is_holiday": _estimate_holiday(date),
                        "lag_1": lag_1,
                        "lag_4": lag_4,
                        "lag_8": lag_8,
                        "rolling_mean_4": rolling_mean_4,
                    }
                ]
            )
            pred = max(0.0, float(model.predict(row)[0]))
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
        rmse = self._train_rmse(sku)

        lower = (forecast - ci_multiplier * rmse).clip(lower=0)
        upper = forecast + ci_multiplier * rmse

        return pd.DataFrame(
            {"forecast": forecast, "lower": lower, "upper": upper},
            index=forecast.index,
        )

    def cross_validate(self, sku: str, n_splits: int = 3) -> CVScore:
        """時系列交差検証でモデル性能を評価する。

        TimeSeriesSplit を使い、データリークなしに RMSE / MAE を計算する。

        Args:
            sku: 対象SKU名。
            n_splits: 分割数。

        Returns:
            CVScore TypedDict（rmse_mean, rmse_std, mae_mean, n_splits）。

        Raises:
            ValueError: 指定SKUのデータが存在しない場合。
        """
        if sku not in self._history:
            raise ValueError(f"SKU '{sku}' のデータが見つかりません。先にfit()を実行してください。")

        featured = _build_features(self._history[sku]).dropna(subset=FEATURE_COLS)
        X = featured[FEATURE_COLS].values
        y = featured["sales"].values

        tscv = TimeSeriesSplit(n_splits=n_splits)
        rmse_scores: list[float] = []
        mae_scores: list[float] = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = lgb.LGBMRegressor(**self._params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            rmse_scores.append(float(np.sqrt(mean_squared_error(y_val, y_pred))))
            mae_scores.append(float(mean_absolute_error(y_val, y_pred)))

        return CVScore(
            sku=sku,
            rmse_mean=float(np.mean(rmse_scores)),
            rmse_std=float(np.std(rmse_scores)),
            mae_mean=float(np.mean(mae_scores)),
            n_splits=n_splits,
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

    def save(self, path: str | Path) -> None:
        """モデルと履歴データを joblib でシリアライズする。

        Args:
            path: 保存先ファイルパス（.pkl 推奨）。
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"models": self._models, "history": self._history, "params": self._params}, path)

    @classmethod
    def load(cls, path: str | Path) -> "DemandForecaster":
        """保存済みモデルを読み込む。

        Args:
            path: モデルファイルパス。

        Returns:
            復元された DemandForecaster インスタンス。
        """
        payload = joblib.load(path)
        instance = cls(lgb_params=payload["params"])
        instance._models = payload["models"]
        instance._history = payload["history"]
        return instance

    def _train_rmse(self, sku: str) -> float:
        """訓練データ上の RMSE を返す（信頼区間推定用）。"""
        history = self._history[sku].copy()
        featured = _build_features(history).dropna(subset=FEATURE_COLS)
        X = featured[FEATURE_COLS]
        y_true = featured["sales"].values
        y_pred = self._models[sku].predict(X)
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
