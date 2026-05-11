# MLOps-пайплайн для классификации мошеннических транзакций

Проект реализует полный цикл ML-системы для датасета Credit Card Fraud Detection:
загрузка данных, DVC-версионирование, Great Expectations validation, обучение моделей,
MLflow tracking, FastAPI serving, Docker и GitHub Actions CI/CD.

## Цели и метрики

Бизнес-цель: минимизировать пропуск мошеннических транзакций.

Основные метрики:
- `Recall` для класса fraud: главный критерий выбора модели.
- `F1-score` для класса fraud: баланс между пропусками и ложными тревогами.
- Дополнительно логируются `ROC-AUC` и `PR-AUC`.

Текущий результат после `dvc repro`:
- best model: `logreg`
- fraud recall: `0.8673`
- fraud F1: `0.1269`

Выбор модели выполняется в `scripts/train.py`: LogisticRegression и RandomForest
логируются как отдельные MLflow runs, затем лучшая модель выбирается по `Recall`
для fraud-класса, с `F1` и `PR-AUC` как дополнительными критериями.

## Структура

- `scripts/load_data.py` - скачивает raw датасет из OpenML.
- `scripts/preprocess.py` - читает ARFF, приводит типы, делает train/test split.
- `scripts/validate_data.py` - Great Expectations validation для train/test.
- `scripts/train.py` - обучает LogisticRegression и RandomForest, логирует в MLflow.
- `dvc.yaml` - DAG `load_data -> preprocess -> validate_data -> train`.
- `app/main.py` - FastAPI service.
- `app/src/model_service.py` - загрузка `models/model.pkl` и inference.
- `app/tests/` - unit и integration tests.
- `.github/workflows/ci.yml` - lint, tests, audit и DVC pipeline.
- `.github/workflows/release.yml` - release image по git-тегам.

## Локальный запуск пайплайна

```bash
cd app
poetry install --only main,pipeline --no-root
cd ..
MLFLOW_TRACKING_URI=file:mlruns dvc repro
```

После выполнения появятся:
- `data/raw/creditcard.arff`
- `data/processed/train.csv`
- `data/processed/test.csv`
- `reports/gx_validation.json`
- `reports/metrics.json`
- `models/model.pkl`
- `models/feature_columns.json`

Данные и модели не коммитятся напрямую, их состояние фиксируется в `dvc.lock`.
Raw, train/test split и модели являются DVC outputs, поэтому их версии
определяются md5-хэшами в `dvc.lock`.

## MLflow

Локальный MLflow stack поднимается через Docker Compose:

```bash
docker compose up -d minio postgres mlflow
```

UI будет доступен на `http://localhost:5000`.

Для логирования в сервер:

```bash
MLFLOW_TRACKING_URI=http://localhost:5000 dvc repro
```

При локальном запуске с `MLFLOW_TRACKING_URI=file:mlruns` результаты runs лежат
в `mlruns/`. В GitHub Actions этот каталог загружается как artifact workflow.

## API

Перед сборкой production image положите модель в контекст `app`:

```bash
mkdir -p app/models
cp models/model.pkl models/feature_columns.json app/models/
docker build --target deploy_build -t fraud-api app
docker run --rm -p 8000:8000 fraud-api
```

Health-check:

```bash
curl http://localhost:8000/health
```

Prediction:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  --data '{"features":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,42]}'
```

## Тесты

```bash
docker build --target test_build -t mlops-app-test app
docker run --rm mlops-app-test sh -lc "flake8 . && ruff check . && pytest && pip-audit"
```

Coverage gate задан в `app/pyproject.toml`: `--cov-fail-under=80`.

## GitHub Actions CI/CD

Pipeline:
- `lint-and-tests`: `flake8`, `ruff`, `pytest` + coverage artifact, matrix Python `3.11`/`3.12`.
- `dependency-audit`: `pip-audit`.
- `dvc-pipeline`: `dvc pull || true`, `dvc repro`, Great Expectations, training, model and MLflow artifacts.
- `release-image`: сборка, smoke-test и публикация Docker image по тегам `v*.*.*`.

Release workflow публикует образы в Docker Hub:

```text
<dockerhub-user-or-org>/mlops-labs:<git-tag>
<dockerhub-user-or-org>/mlops-labs:latest
```

Для публикации нужно добавить GitHub repository secrets:
- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

Опционально можно задать repository variable:
- `DOCKERHUB_REPOSITORY`, например `artgth/mlops-labs`

Если `DOCKERHUB_REPOSITORY` не задан, workflow использует:

```text
${DOCKERHUB_USERNAME}/mlops-labs
```

Release job перед сборкой выполняет `dvc repro`, копирует модель в `app/models`,
собирает image, выполняет smoke-test `/health` и `/predict`, затем пушит image.
