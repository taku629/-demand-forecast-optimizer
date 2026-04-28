FROM python:3.11-slim

WORKDIR /app

# 依存ライブラリを先にコピー（レイヤーキャッシュ最適化）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# サンプルデータを生成
RUN python data/generate_sample_data.py

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true"]
