"""需要予測 × 在庫最適化 Streamlit アプリ。

総合商社の物流・在庫管理業務を想定した需要予測と数理最適化の統合デモ。
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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
    df = pd.read_json(io.StringIO(df_json), orient="records")
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
    warehouse_capacity = st.number_input(
        "倉庫容量 (個)", min_value=50, max_value=2000, value=500, step=50
    )
    min_order_lot = st.number_input(
        "最小発注ロット (個)", min_value=1, max_value=200, value=20, step=5
    )

    st.markdown("---")
    st.info(
        "**使い方**\n\n"
        "パラメータを変更すると自動的に需要予測と最適発注スケジュールが再計算されます。"
    )


# ─── 予測・最適化の実行（一度だけ呼ぶ） ─────────────────────────────────────────

history_df = forecaster.get_history(selected_sku)
history_df["date"] = pd.to_datetime(history_df["date"])
history_df = history_df.sort_values("date")

# predict_with_confidence が内部で predict を呼ぶので、forecast から demand_series を取得
forecast_df = forecaster.predict_with_confidence(selected_sku, n_weeks=4)
demand_series = forecast_df["forecast"]

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
    st.metric("4週間予測需要合計", f"{int(demand_series.sum()):,} 個")
with col2:
    st.metric("推奨発注量合計", f"{sum(opt_result['order_schedule']):,} 個")
with col3:
    st.metric("最適化総コスト", format_currency(opt_result["total_cost"]))
with col4:
    n_orders = sum(1 for o in opt_result["order_schedule"] if o > 0)
    st.metric("発注回数", f"{n_orders} 回 / 4週")

st.markdown("---")


# ─── タブ構成 ────────────────────────────────────────────────────────────────

tab_main, tab_scenario, tab_model = st.tabs(
    ["📊 メインダッシュボード", "🔄 シナリオ比較", "🤖 モデル詳細"]
)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1: メインダッシュボード
# ════════════════════════════════════════════════════════════════════════════════

with tab_main:

    # ── グラフ1：需要予測 ──────────────────────────────────────────────────────

    st.subheader("📈 需要予測グラフ")
    recent = history_df.tail(16)
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(
        x=recent["date"], y=recent["sales"],
        name="過去実績", mode="lines+markers",
        line=dict(color="#2196F3", width=2), marker=dict(size=5),
    ))
    fig_forecast.add_trace(go.Scatter(
        x=forecast_df.index, y=forecast_df["forecast"],
        name="4週間予測", mode="lines+markers",
        line=dict(color="#F44336", width=2, dash="dash"),
        marker=dict(size=8, symbol="diamond"),
    ))
    fig_forecast.add_trace(go.Scatter(
        x=list(forecast_df.index) + list(forecast_df.index[::-1]),
        y=list(forecast_df["upper"]) + list(forecast_df["lower"][::-1]),
        fill="toself", fillcolor="rgba(244,67,54,0.1)",
        line=dict(color="rgba(255,255,255,0)"),
        name="95% 信頼区間",
    ))
    fig_forecast.update_layout(
        xaxis_title="日付", yaxis_title="販売数 (個)", height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    st.plotly_chart(fig_forecast, use_container_width=True)

    # ── グラフ2 & 3 ───────────────────────────────────────────────────────────

    col_l, col_r = st.columns(2)
    date_labels = [pd.Timestamp(d).strftime("%m/%d") for d in opt_result["dates"]]

    with col_l:
        st.subheader("📦 最適発注スケジュール")
        fig_order = go.Figure(go.Bar(
            x=date_labels, y=opt_result["order_schedule"],
            marker_color=["#4CAF50" if o > 0 else "#E0E0E0" for o in opt_result["order_schedule"]],
            text=opt_result["order_schedule"], textposition="auto",
        ))
        fig_order.update_layout(xaxis_title="週", yaxis_title="発注量 (個)", height=330, showlegend=False)
        st.plotly_chart(fig_order, use_container_width=True)

    with col_r:
        st.subheader("🏭 在庫推移シミュレーション")
        safety_stock = int(demand_series.mean() * 1.5)
        fig_stock = go.Figure()
        fig_stock.add_trace(go.Scatter(
            x=date_labels, y=opt_result["stock_levels"],
            name="在庫量", mode="lines+markers",
            line=dict(color="#9C27B0", width=2), marker=dict(size=8),
            fill="tozeroy", fillcolor="rgba(156,39,176,0.1)",
        ))
        fig_stock.add_trace(go.Scatter(
            x=date_labels, y=[int(d) for d in demand_series.values],
            name="予測需要", mode="lines+markers",
            line=dict(color="#FF9800", width=2, dash="dot"),
        ))
        fig_stock.add_hline(
            y=safety_stock, line_dash="dash", line_color="red",
            annotation_text=f"安全在庫: {safety_stock}個",
            annotation_position="right",
        )
        fig_stock.update_layout(
            xaxis_title="週", yaxis_title="在庫量 (個)", height=330,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            hovermode="x unified",
        )
        st.plotly_chart(fig_stock, use_container_width=True)

    # ── グラフ4：コスト内訳 ───────────────────────────────────────────────────

    st.subheader("💰 コスト内訳")
    breakdown = opt_result["cost_breakdown"]
    fig_cost = go.Figure(go.Bar(
        x=["保管コスト", "発注コスト", "欠品ペナルティ"],
        y=[breakdown["holding"], breakdown["ordering"], breakdown["stockout"]],
        marker_color=["#2196F3", "#4CAF50", "#F44336"],
        text=[format_currency(v) for v in breakdown.values()],
        textposition="auto",
    ))
    fig_cost.update_layout(yaxis_title="コスト (円)", height=300, showlegend=False)
    st.plotly_chart(fig_cost, use_container_width=True)

    # ── テーブル5：発注推奨テーブル + ダウンロード ────────────────────────────

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

    csv_bytes = rec_table.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        label="📥 発注推奨テーブルをCSVでダウンロード",
        data=csv_bytes,
        file_name=f"order_schedule_{selected_sku}.csv",
        mime="text/csv",
    )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2: シナリオ比較
# ════════════════════════════════════════════════════════════════════════════════

with tab_scenario:
    st.subheader("🔄 コストパラメータ感度分析")
    st.caption("欠品ペナルティを変化させたときの総コストと発注回数の変化を比較します。")

    penalties = [100, 300, 500, 1_000, 2_000, 5_000]
    scenario_records = []
    for penalty in penalties:
        res = optimizer.optimize(
            demand_forecast=demand_series,
            current_stock=current_stock,
            holding_cost=float(holding_cost),
            order_fixed_cost=float(order_fixed_cost),
            stockout_penalty=float(penalty),
            warehouse_capacity=int(warehouse_capacity),
            min_order_lot=int(min_order_lot),
        )
        scenario_records.append({
            "欠品ペナルティ (円)": penalty,
            "総コスト (円)": res["total_cost"],
            "発注回数": sum(1 for o in res["order_schedule"] if o > 0),
            "発注量合計 (個)": sum(res["order_schedule"]),
            "保管コスト (円)": res["cost_breakdown"]["holding"],
            "発注コスト (円)": res["cost_breakdown"]["ordering"],
            "欠品コスト (円)": res["cost_breakdown"]["stockout"],
        })
    scenario_df = pd.DataFrame(scenario_records)

    fig_scenario = go.Figure()
    fig_scenario.add_trace(go.Scatter(
        x=scenario_df["欠品ペナルティ (円)"],
        y=scenario_df["総コスト (円)"],
        name="総コスト", mode="lines+markers",
        line=dict(color="#E91E63", width=2), marker=dict(size=8),
        yaxis="y1",
    ))
    fig_scenario.add_trace(go.Bar(
        x=scenario_df["欠品ペナルティ (円)"],
        y=scenario_df["発注量合計 (個)"],
        name="発注量合計 (個)", marker_color="rgba(33,150,243,0.4)",
        yaxis="y2",
    ))
    fig_scenario.update_layout(
        xaxis_title="欠品ペナルティ (円/個)",
        yaxis=dict(title="総コスト (円)", side="left"),
        yaxis2=dict(title="発注量合計 (個)", side="right", overlaying="y"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=380,
        hovermode="x unified",
    )
    st.plotly_chart(fig_scenario, use_container_width=True)

    st.dataframe(
        scenario_df.style.format({
            "総コスト (円)": "¥{:,.0f}",
            "保管コスト (円)": "¥{:,.0f}",
            "発注コスト (円)": "¥{:,.0f}",
            "欠品コスト (円)": "¥{:,.0f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # ── 在庫初期値の感度分析 ─────────────────────────────────────────────────

    st.subheader("📦 現在庫数感度分析")
    st.caption("現在庫数を変化させたときのコスト影響を確認します。")

    stock_levels_range = list(range(0, 301, 50))
    stock_records = []
    for init_stock in stock_levels_range:
        res = optimizer.optimize(
            demand_forecast=demand_series,
            current_stock=init_stock,
            holding_cost=float(holding_cost),
            order_fixed_cost=float(order_fixed_cost),
            stockout_penalty=float(stockout_penalty),
            warehouse_capacity=int(warehouse_capacity),
            min_order_lot=int(min_order_lot),
        )
        stock_records.append({
            "現在庫数 (個)": init_stock,
            "総コスト (円)": res["total_cost"],
            "発注量合計 (個)": sum(res["order_schedule"]),
        })
    stock_df = pd.DataFrame(stock_records)

    fig_stock_sens = go.Figure()
    fig_stock_sens.add_trace(go.Scatter(
        x=stock_df["現在庫数 (個)"], y=stock_df["総コスト (円)"],
        mode="lines+markers", name="総コスト",
        line=dict(color="#FF5722", width=2), marker=dict(size=8),
    ))
    fig_stock_sens.add_vline(
        x=current_stock, line_dash="dash", line_color="blue",
        annotation_text=f"現在設定: {current_stock}個",
    )
    fig_stock_sens.update_layout(
        xaxis_title="現在庫数 (個)", yaxis_title="総コスト (円)", height=320
    )
    st.plotly_chart(fig_stock_sens, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3: モデル詳細
# ════════════════════════════════════════════════════════════════════════════════

with tab_model:

    # ── 特徴量重要度 ──────────────────────────────────────────────────────────

    st.subheader("🔍 特徴量重要度 (LightGBM)")
    feat_imp_df = forecaster.get_feature_importance()
    sku_imp = feat_imp_df[feat_imp_df["sku"] == selected_sku].sort_values("importance")
    fig_imp = go.Figure(go.Bar(
        x=sku_imp["importance"], y=sku_imp["feature"],
        orientation="h", marker_color="#00BCD4",
        text=sku_imp["importance"].astype(int), textposition="auto",
    ))
    fig_imp.update_layout(xaxis_title="重要度スコア", height=320, showlegend=False)
    st.plotly_chart(fig_imp, use_container_width=True)

    # ── 交差検証スコア ────────────────────────────────────────────────────────

    st.subheader("📐 時系列交差検証スコア (TimeSeriesSplit, n=3)")

    @st.cache_data
    def compute_cv_scores(df_json: str) -> pd.DataFrame:
        """全SKUのCVスコアを計算する（キャッシュ付き）。"""
        df = pd.read_json(io.StringIO(df_json), orient="records")
        df["date"] = pd.to_datetime(df["date"])
        fc = DemandForecaster()
        fc.fit(df)
        records = []
        for sku in sorted(df["sku"].unique()):
            score = fc.cross_validate(sku, n_splits=3)
            records.append({
                "SKU": sku,
                "RMSE (平均)": f"{score['rmse_mean']:.1f}",
                "RMSE (標準偏差)": f"{score['rmse_std']:.1f}",
                "MAE (平均)": f"{score['mae_mean']:.1f}",
            })
        return pd.DataFrame(records)

    with st.spinner("交差検証スコアを計算中..."):
        cv_df = compute_cv_scores(df_all.to_json(orient="records"))

    st.dataframe(cv_df, use_container_width=True, hide_index=True)
    st.caption("RMSE・MAEは販売数（個）単位。低いほどモデルの汎化性能が高い。")

    # ── 全SKU予測オーバーレイ ─────────────────────────────────────────────────

    st.subheader("🌐 全SKU需要予測比較")
    fig_all = go.Figure()
    colors = ["#2196F3", "#4CAF50", "#F44336", "#FF9800", "#9C27B0"]
    for i, sku in enumerate(skus):
        fc_series = forecaster.predict(sku, n_weeks=4)
        hist = forecaster.get_history(sku)
        hist["date"] = pd.to_datetime(hist["date"])
        recent_hist = hist.sort_values("date").tail(8)
        fig_all.add_trace(go.Scatter(
            x=recent_hist["date"], y=recent_hist["sales"],
            name=f"{sku} (実績)", line=dict(color=colors[i], width=1, dash="dot"),
            opacity=0.6, showlegend=True,
        ))
        fig_all.add_trace(go.Scatter(
            x=fc_series.index, y=fc_series.values,
            name=f"{sku} (予測)", mode="lines+markers",
            line=dict(color=colors[i], width=2), marker=dict(size=7),
        ))
    fig_all.update_layout(
        xaxis_title="日付", yaxis_title="販売数 (個)", height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )
    st.plotly_chart(fig_all, use_container_width=True)


# ─── フッター ────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    f"最適化ステータス: `{opt_result['status']}` ｜ "
    f"対象SKU: {selected_sku} ｜ "
    "ポートフォリオ用デモアプリ — 丸紅インターン向け"
)
