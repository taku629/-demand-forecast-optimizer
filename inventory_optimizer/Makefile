.PHONY: help install data lint typecheck test run docker-build docker-run clean

PYTHON := python
APP_DIR := $(shell pwd)

help:  ## このヘルプを表示
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:  ## 依存ライブラリをインストール
	pip install -r requirements.txt
	pip install pytest pytest-cov ruff mypy pandas-stubs

data:  ## サンプルデータを生成
	$(PYTHON) data/generate_sample_data.py

lint:  ## ruff でコードチェック
	ruff check src/ tests/ app.py

lint-fix:  ## ruff で自動修正
	ruff check --fix src/ tests/ app.py

typecheck:  ## mypy で型チェック
	mypy src/ --ignore-missing-imports

test:  ## ユニットテストを実行
	pytest tests/ -v

test-cov:  ## カバレッジ付きテスト
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

run:  ## Streamlit アプリを起動
	streamlit run app.py

docker-build:  ## Docker イメージをビルド
	docker build -t inventory-optimizer .

docker-run:  ## Docker コンテナを起動
	docker run -p 8501:8501 inventory-optimizer

clean:  ## キャッシュ・一時ファイルを削除
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage coverage.xml
