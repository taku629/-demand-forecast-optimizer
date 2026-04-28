"""サンプル販売データ生成スクリプト。

SKU 5種類の過去52週分の週次販売データを生成し、
data/sales_history.csv として保存する。
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_sales_data(
    n_weeks: int = 52,
    skus: list[str] | None = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """過去n_weeks週分の週次販売データを生成する。

    Args:
        n_weeks: 生成する週数。
        skus: SKUリスト。Noneの場合は商品A〜Eを使用。
        random_seed: 乱数シード。

    Returns:
        date, sku, sales, price, is_holiday カラムを持つDataFrame。
    """
    if skus is None:
        skus = ["商品A", "商品B", "商品C", "商品D", "商品E"]

    rng = np.random.default_rng(random_seed)
    end_date = pd.Timestamp("2024-12-29")  # 日曜日終わり
    dates = pd.date_range(end=end_date, periods=n_weeks, freq="W-SUN")

    # SKUごとのパラメータ
    sku_params = {
        "商品A": {"base": 100, "trend": 1.5, "amplitude": 20, "phase": 0, "price": 1200},
        "商品B": {"base": 200, "trend": -0.5, "amplitude": 40, "phase": np.pi / 3, "price": 800},
        "商品C": {"base": 50, "trend": 2.0, "amplitude": 10, "phase": np.pi / 2, "price": 3500},
        "商品D": {"base": 150, "trend": 0.8, "amplitude": 30, "phase": np.pi, "price": 600},
        "商品E": {"base": 80, "trend": 1.2, "amplitude": 15, "phase": np.pi * 1.5, "price": 2200},
    }

    # 祝日フラグ（正月・GW・お盆・クリスマス周辺）
    holiday_weeks = set()
    for date in dates:
        month, week = date.month, date.isocalendar().week
        if month == 1 and week <= 2:  # 正月
            holiday_weeks.add(date)
        elif month in (4, 5) and 17 <= week <= 19:  # GW
            holiday_weeks.add(date)
        elif month == 8 and 32 <= week <= 34:  # お盆
            holiday_weeks.add(date)
        elif month == 12 and week >= 51:  # 年末年始
            holiday_weeks.add(date)

    records = []
    for sku in skus:
        params = sku_params[sku]
        for i, date in enumerate(dates):
            # トレンド成分
            trend = params["trend"] * i
            # 季節性成分（52週周期）
            seasonality = params["amplitude"] * np.sin(
                2 * np.pi * i / 52 + params["phase"]
            )
            # 祝日効果
            holiday_boost = 15 if date in holiday_weeks else 0
            is_holiday = 1 if date in holiday_weeks else 0
            # ランダムノイズ
            noise = rng.normal(0, params["base"] * 0.08)
            # 販売数（最小0に丸め）
            sales = max(0, round(params["base"] + trend + seasonality + holiday_boost + noise))

            records.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "sku": sku,
                    "sales": int(sales),
                    "price": params["price"],
                    "is_holiday": is_holiday,
                }
            )

    return pd.DataFrame(records)


def main() -> None:
    """サンプルデータを生成してCSVに保存する。"""
    output_dir = Path(__file__).parent
    output_path = output_dir / "sales_history.csv"

    df = generate_sales_data()
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"データを保存しました: {output_path}")
    print(f"レコード数: {len(df)}")
    print(f"SKU: {df['sku'].unique().tolist()}")
    print(f"期間: {df['date'].min()} 〜 {df['date'].max()}")
    print(df.head(10).to_string())


if __name__ == "__main__":
    main()
