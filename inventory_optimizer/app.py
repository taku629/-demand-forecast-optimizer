"""需要予測 × 在庫最適化 Streamlit アプリ。

総合商社の物流・在庫管理業務を想定した需要予測と数理最適化の統合デモ。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# パスを src に通す
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.forecast import DemandForecaster
from src.optimizer import InventoryOptimizer
from src.utils import build_recommendation_table, format_currency, load_sales_data


# ─── ページ設定 ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="需要予測 × 在庫最適化",
    page_icon="📦",
    layout="wide",
)

st.title("📦 需要予測 × 在庫最適化システム")
st.caption("総合商社向け在庫マネジメント支援ツール | LightGBM × PuLP 整数線形計画")


# ─── データ読み込み・モデル学習（キャッシュ） ──────────────────────────────────

@st.cache_data
def load_data() -> pd.DataFrame:
    """販売データを読み込む（キャッシュ付き）。"""
    data_path = Path(__file__).parent / "data" / "sales_history.csv"
    if not data_path.exists():
        from data.generate_sample_data import generate_sales_data
        df = generate_sales_data()
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_path, index=False, encoding="utf-8-sig")
        return df
    return load_sales_data(data_path)


@st.cache_resource
def train_forecaster(df_json: str) -> DemandForecaster:
    """需要予測モデルを学習する（リソースキャッシュ）。"""
    df = pd.read_json(df_json, orient="records")
    df["date"] = pd.to_datetime(df["date"])
    forecaster = DemandForecaster()
    forecaster.fit(df)
    return forecaster


with st.spinner("データ読み込み・モデル学習中..."):
    df_all = load_data()
    forecaster = train_forecaster(df_all.to_json(orient="records"))

skus = sorted(df_all["sku"].unique().tolist())


# ─── サイドバー ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ パラメータ設定")

    selected_sku = st.selectbox("SKU選択", skus, index=0)

    st.subheader("在庫・コスト設定")
    current_stock = st.slider("現在庫数 (個)", min_value=0, max_value=500, value=100, step=10)
    holding_cost = st.slider("保管コスト (円/個/週)", min_value=1, max_value=100, value=10)
    order_fixed_cost = st.slider(
        "発注固定費 (円/回)", min_value=1_000, max_value=50_000, value=10_000, step=1_000
    )
    stockout_penalty = st.slider(
        "欠品ペナルティ (円/個)", min_value=100, max_value=5_000, value=500, step=100
    )

    st.subheader("倉庫設定")
    warehouse_capacity = st.number_input("倉庫容量 (個)", min_value=50, max_value=2000, value=500, step=50)
    min_order_lot = st.number_input("最小発注ロット (個)", min_value=1, max_value=200, value=20, step=5)

    st.markdown("---")
    st.info(
        "**使い方**\n\n"
        "パラメータを変更すると自動的に需要予測と最適発注スケジュールが再計算されます。"
    )


# ─── 予測・最適化の実行 ────────────────────────────────────────────────────────

history_df = forecaster.get_history(selected_sku)
history_df["date"] = pd.to_datetime(history_df["date"])
history_df = history_df.sort_values("date")

forecast_df = forecaster.predict_with_confidence(selected_sku, n_weeks=4)
demand_series = forecaster.predict(selected_sku, n_weeks=4)

optimizer = InventoryOptimizer()
opt_result = optimizer.optimize(
    demand_forecast=demand_series,
    current_stock=current_stock,
    holding_cost=float(holding_cost),
    order_fixed_cost=float(order_fixed_cost),
    stockout_penalty=float(stockout_penalty),
    warehouse_capacity=int(warehouse_capacity),
    min_order_lot=int(min_order_lot),
)


# ─── KPIカード ────────────────────────────────────────────────────────────────

col1, col2, col3, col4 = st.columns(4)
with col1:
    total_demand = int(demand_series.sum())
    st.metric("4週間予測需要合計", f"{total_demand:,} 個")
with col2:
    total_order = sum(opt_result["order_schedule"])
    st.metric("推奨発注量合計", f"{total_order:,} 個")
with col3:
    st.metric("最適化総コスト", format_currency(opt_result["total_cost"]))
with col4:
    n_orders = sum(1 for o in opt_result["order_schedule"] if o > 0)
    st.metric("発注回数", f"{n_orders} 回 / 4週")


st.markdown("---")


# ─── グラフ1：需要予測 ────────────────────────────────────────────────────────

st.subheader("📈 需要予測グラフ")

fig_forecast = go.Figure()

# 過去実績（直近16週）
recent = history_df.tail(16)
fig_forecast.add_trace(
    go.Scatter(
        x=recent["date"],
        y=recent["sales"],
        name="過去実績",
        line=dict(color="#2196F3", width=2),
        mode="lines+markers",
        marker=dict(size=5),
    )
)

# 予測値（赤破線）
fig_forecast.add_trace(
    go.Scatter(
        x=forecast_df.index,
        y=forecast_df["forecast"],
        name="4週間予測",
        line=dict(color="#F44336", width=2, dash="dash"),
        mode="lines+markers",
        marker=dict(size=8, symbol="diamond"),
    )
)

# 信頼区間
fig_forecast.add_trace(
    go.Scatter(
        x=list(forecast_df.index) + list(forecast_df.index[::-1]),
        y=list(forecast_df["upper"]) + list(forecast_df["lower"][::-1]),
        fill="toself",
        fillcolor="rgba(244,67,54,0.1)",
        line=dict(color="rgba(255,255,255,0)"),
        name="95% 信頼区間",
        showlegend=True,
    )
)

fig_forecast.update_layout(
    xaxis_title="日付",
    yaxis_title="販売数 (個)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
    height=380,
)
st.plotly_chart(fig_forecast, use_container_width=True)


# ─── グラフ2 & 3：発注スケジュール + 在庫推移 ──────────────────────────────────

col_l, col_r = st.columns(2)

with col_l:
    st.subheader("📦 最適発注スケジュール")
    date_labels = [pd.Timestamp(d).strftime("%m/%d") for d in opt_result["dates"]]

    fig_order = go.Figure(
        go.Bar(
            x=date_labels,
            y=opt_result["order_schedule"],
            marker_color=["#4CAF50" if o > 0 else "#E0E0E0" for o in opt_result["order_schedule"]],
            text=opt_result["order_schedule"],
            textposition="auto",
            name="発注量",
        )
    )
    fig_order.update_layout(
        xaxis_title="週",
        yaxis_title="発注量 (個)",
        height=330,
        showlegend=False,
    )
    st.plotly_chart(fig_order, use_container_width=True)

with col_r:
    st.subheader("🏭 在庫推移シミュレーション")
    safety_stock = int(demand_series.mean() * 1.5)  # 安全在庫ライン（1.5週分）

    fig_stock = go.Figure()
    fig_stock.add_trace(
        go.Scatter(
            x=date_labels,
            y=opt_result["stock_levels"],
            name="在庫量",
            line=dict(color="#9C27B0", width=2),
            mode="lines+markers",
            marker=dict(size=8),
            fill="tozeroy",
            fillcolor="rgba(156,39,176,0.1)",
        )
    )
    fig_stock.add_trace(
        go.Scatter(
            x=date_labels,
            y=[int(d) for d in demand_series.values],
            name="予測需要",
            line=dict(color="#FF9800", width=2, dash="dot"),
            mode="lines+markers",
        )
    )
    fig_stock.add_hline(
        y=safety_stock,
        line_dash="dash",
        line_color="red",
        annotation_text=f"安全在庫: {safety_stock}個",
        annotation_position="right",
    )
    fig_stock.update_layout(
        xaxis_title="週",
        yaxis_title="在庫量 (個)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=330,
        hovermode="x unified",
    )
    st.plotly_chart(fig_stock, use_container_width=True)


# ─── グラフ4：コスト内訳 ──────────────────────────────────────────────────────

st.subheader("💰 コスト内訳")
breakdown = opt_result["cost_breakdown"]
fig_cost = go.Figure(
    go.Bar(
        x=["保管コスト", "発注コスト", "欠品ペナルティ"],
        y=[breakdown["holding"], breakdown["ordering"], breakdown["stockout"]],
        marker_color=["#2196F3", "#4CAF50", "#F44336"],
        text=[format_currency(v) for v in [breakdown["holding"], breakdown["ordering"], breakdown["stockout"]]],
        textposition="auto",
    )
)
fig_cost.update_layout(
    yaxis_title="コスト (円)",
    height=300,
    showlegend=False,
)
st.plotly_chart(fig_cost, use_container_width=True)


# ─── テーブル5：発注推奨テーブル ─────────────────────────────────────────────

st.subheader("📋 発注推奨テーブル")
rec_table = build_recommendation_table(
    dates=opt_result["dates"],
    order_schedule=opt_result["order_schedule"],
    demand_forecast=demand_series.values.tolist(),
    stock_levels=opt_result["stock_levels"],
    holding_cost=float(holding_cost),
    order_fixed_cost=float(order_fixed_cost),
)
st.dataframe(rec_table, use_container_width=True, hide_index=True)


# ─── グラフ6：特徴量重要度 ───────────────────────────────────────────────────

st.subheader("🔍 特徴量重要度 (LightGBM)")
feat_imp_df = forecaster.get_feature_importance()
sku_imp = feat_imp_df[feat_imp_df["sku"] == selected_sku].sort_values("importance")

fig_imp = go.Figure(
    go.Bar(
        x=sku_imp["importance"],
        y=sku_imp["feature"],
        orientation="h",
        marker_color="#00BCD4",
        text=sku_imp["importance"].astype(int),
        textposition="auto",
    )
)
fig_imp.update_layout(
    xaxis_title="重要度スコア",
    yaxis_title="特徴量",
    height=320,
    showlegend=False,
)
st.plotly_chart(fig_imp, use_container_width=True)


# ─── フッター ────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    f"最適化ステータス: `{opt_result['status']}` ｜ "
    f"対象SKU: {selected_sku} ｜ "
    "ポートフォリオ用デモアプリ — 丸紅インターン向け"
)
