# 需要予測 × 在庫最適化システム

総合商社（丸紅）の物流・在庫管理業務を想定した、機械学習と数理最適化を組み合わせたWebアプリです。

## プロジェクト概要

多品種・多拠点の在庫管理は総合商社における重要な業務課題です。  
本システムは以下の2段階アプローチでこの課題に取り組みます：

1. **需要予測**（LightGBM）: 過去52週の販売実績から将来4週間の需要を予測
2. **最適発注計画**（PuLP / 整数線形計画）: 予測需要をもとに、保管・発注・欠品コストを最小化する発注スケジュールを算出

## アーキテクチャ

```mermaid
flowchart LR
    A[📊 過去販売データ\nCSV生成] -->|52週分\n5SKU| B[🤖 需要予測\nLightGBM]
    B -->|4週間予測値\n+信頼区間| C[⚙️ 在庫最適化\nPuLP / 整数線形計画]
    C -->|最適発注スケジュール\nコスト内訳| D[🖥️ Streamlit UI\nPlotly可視化]
    D --> E[📋 発注推奨テーブル\n意思決定支援]

    style A fill:#E3F2FD
    style B fill:#E8F5E9
    style C fill:#FFF3E0
    style D fill:#F3E5F5
    style E fill:#FCE4EC
```

## セットアップ手順

```bash
# 1. 依存ライブラリのインストール
pip install -r requirements.txt

# 2. サンプルデータの生成
python data/generate_sample_data.py

# 3. アプリの起動
streamlit run app.py
```

ブラウザで `http://localhost:8501` が自動的に開きます。

## ディレクトリ構成

```
inventory_optimizer/
├── README.md
├── requirements.txt
├── data/
│   ├── generate_sample_data.py   # SKU 5種類×52週のデータ生成
│   └── sales_history.csv         # 生成済みサンプルデータ
├── src/
│   ├── forecast.py               # LightGBM需要予測モジュール
│   ├── optimizer.py              # PuLP在庫最適化モジュール
│   └── utils.py                  # データ読み込み・フォーマット
├── tests/
│   ├── test_forecast.py          # 需要予測ユニットテスト
│   └── test_optimizer.py         # 最適化ユニットテスト
└── app.py                        # Streamlitメインアプリ
```

## 技術的工夫ポイント

- **時系列を考慮した特徴量エンジニアリング**: ラグ特徴量（lag_1/4/8週）と移動平均（rolling_mean_4）を用いてデータリークなしに未来を予測。将来週の推論時は予測値を逐次フィードバックする再帰的予測を実装。

- **不確実性の定量化**: 訓練残差のRMSEから95%信頼区間を計算し、予測の信頼性を可視化。意思決定者が「ベストケース / ワーストケース」を把握できる設計。

- **整数線形計画による実用的な最適化**: 最小発注ロット制約をBig-M法でモデル化し、離散的な発注決定を整数変数で厳密に扱う。欠品ペナルティ・保管コスト・発注固定費のトレードオフをパラメータ感度分析で即座に確認可能。

- **SKUごとのモデル管理**: 商品特性（トレンド・季節性・価格帯）が異なる各SKUに対し独立したLightGBMモデルを管理。`@st.cache_resource`でセッション間のモデル再利用を実現し、UXを向上。

- **ロバストな最適化フォールバック**: PuLPが実行不可能解を返す極端なパラメータ設定時にも、安全在庫ベースの発注スケジュールにフォールバックしてアプリが落ちない設計。インターンデモ中の突発的パラメータ変更に対応。

## テスト実行

```bash
pytest tests/ -v
```

## ライセンス

MIT License
