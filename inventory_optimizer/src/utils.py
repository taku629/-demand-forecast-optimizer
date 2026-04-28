"""ユーティリティ関数モジュール。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_sales_data(csv_path: str | Path | None = None) -> pd.DataFrame:
    """販売履歴CSVを読み込む。

    Args:
        csv_path: CSVファイルパス。Noneの場合はデフォルトパスを使用。

    Returns:
        date, sku, sales, price, is_holiday カラムを持つDataFrame。

    Raises:
        FileNotFoundError: CSVファイルが存在しない場合。
    """
    if csv_path is None:
        csv_path = Path(__file__).parent.parent / "data" / "sales_history.csv"
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"販売データが見つかりません: {path}\n"
            "先に `python data/generate_sample_data.py` を実行してください。"
        )
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    return df


def format_currency(value: float) -> str:
    """金額を日本円フォーマットで返す。

    Args:
        value: 金額（円）。

    Returns:
        フォーマット済み文字列（例: "¥12,345"）。
    """
    return f"¥{value:,.0f}"


def build_recommendation_table(
    dates: list,
    order_schedule: list[int],
    demand_forecast: list[float],
    stock_levels: list[int],
    holding_cost: float,
    order_fixed_cost: float,
) -> pd.DataFrame:
    """発注推奨テーブルを構築する。

    Args:
        dates: 日付リスト。
        order_schedule: 各週の発注量リスト。
        demand_forecast: 各週の予測需要リスト。
        stock_levels: 各週末の在庫量リスト。
        holding_cost: 1個あたり週次保管コスト（円）。
        order_fixed_cost: 1回の発注固定費（円）。

    Returns:
        週 / 発注量 / 予測需要 / 予想在庫 / コスト カラムを持つDataFrame。
    """
    records = []
    for i, date in enumerate(dates):
        order = order_schedule[i]
        demand = demand_forecast[i]
        stock = stock_levels[i]
        cost = holding_cost * stock + (order_fixed_cost if order > 0 else 0)
        records.append(
            {
                "週": pd.Timestamp(date).strftime("%Y-%m-%d"),
                "発注量 (個)": order,
                "予測需要 (個)": round(demand),
                "予想在庫 (個)": stock,
                "週次コスト (円)": f"¥{cost:,.0f}",
            }
        )
    return pd.DataFrame(records)
