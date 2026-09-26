# MIND News Recommender

A modular, explainable content-based news recommendation system built around the Microsoft News Dataset (MINDlarge). The primary work is the ML pipeline: sparse TF-IDF title features, user-profile aggregation, explicit cosine similarity, offline evaluation, and comparison with a popularity baseline. FastAPI, PostgreSQL, and Streamlit provide the serving and demonstration layer.

## Project layout

```text
config/          Settings and async database lifecycle
models/          SQLAlchemy entities
repositories/    Async persistence boundaries
schemas/         Pydantic API contracts
services/        ML engine and business orchestration
routers/         Health and inference endpoints
notebooks/       Offline dataset and model training pipeline
streamlit_app.py Streamlit frontend
tests/           Unit and route contract tests
AI_ML_GUIDE.md   Concepts and metrics
data/artifacts/ Serialized model.pkl output
data/            Dataset and local runtime storage
```

## Local setup

The dependency pins support Python 3.10 and Python 3.11. If an existing virtual environment was created with incompatible packages, recreate it before installing:

```bash
cd /home/ubuntu/mind-news-recommender
deactivate 2>/dev/null || true
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
python --version
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter notebook notebooks/model_training.ipynb
```

Create the PostgreSQL database before starting the API:

```bash
createdb news_recommender
cp .env.example .env
# Edit DATABASE_URL with your local PostgreSQL credentials.
```

Place your raw MINDlarge folders into the project using this layout:

```text
data/raw/MINDlarge_train/{news.tsv,behaviors.tsv}
data/raw/MINDlarge_dev/{news.tsv,behaviors.tsv}
data/raw/MINDlarge_test/{news.tsv,behaviors.tsv}
```

Then run all notebook cells. The notebook trains TF-IDF title vectors, compares them with a popularity baseline, evaluates Precision@10, Recall@10, Hit Rate@10, and MRR@10 on development impressions, and writes the trained artifact to `data/artifacts/model.pkl`.

Always train and serve the artifact from the same virtual environment. If startup reports a NumPy pickle error such as `No module named numpy._core.numeric`, remove the artifact and retrain it with the active environment:

```bash
rm -f data/artifacts/model.pkl
python -m ipykernel install --user --name mind-recommender --display-name "Python (mind-recommender)"
jupyter notebook notebooks/model_training.ipynb
```

In Jupyter, select **Python (mind-recommender)** as the kernel, then run every cell from the beginning.

Run automated tests:

```bash
pytest -q
```

Start the API from the project root:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

In a second terminal, launch the frontend:

```bash
streamlit run streamlit_app.py
```

Health endpoints:

```bash
curl http://localhost:8000/api/v1/live
curl http://localhost:8000/api/v1/ready
```

Recommendation request:

```bash
curl -X POST http://localhost:8000/api/v1/recommend \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"demo-user","click_history":["N12345","N67890"],"top_k":5}'
```

Build and run the production image after training the artifact:

```bash
docker build -t mind-news-recommender .
docker run --rm -p 8000:8000 mind-news-recommender
```
