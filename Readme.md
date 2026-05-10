# MLOps-пайплайн для классификации мошеннических транзакций

Проект реализует полный цикл ML-системы для датасета Credit Card Fraud Detection:
загрузка данных, DVC-версионирование, Great Expectations validation, обучение моделей,
MLflow tracking, FastAPI serving, Docker и GitLab CI/CD.

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

## Структура

- `scripts/load_data.py` - скачивает raw датасет из OpenML.
- `scripts/preprocess.py` - читает ARFF, приводит типы, делает train/test split.
- `scripts/validate_data.py` - Great Expectations validation для train/test.
- `scripts/train.py` - обучает LogisticRegression и RandomForest, логирует в MLflow.
- `dvc.yaml` - DAG `load_data -> preprocess -> validate_data -> train`.
- `app/main.py` - FastAPI service.
- `app/src/model_service.py` - загрузка `models/model.pkl` и inference.
- `app/tests/` - unit и integration tests.
- `.gitlab-ci.yml` - GitLab CI/CD.

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

## GitLab CI/CD

Pipeline:
- `lint`: `flake8` и `ruff`, matrix Python `3.11`/`3.12`.
- `tests`: `pytest` + coverage artifact, matrix Python `3.11`/`3.12`.
- `dependency_audit`: `pip-audit`.
- `dvc_pipeline`: `dvc pull || dvc repro`, Great Expectations, training, model artifacts.
- `release_image`: сборка и публикация Docker image по тегам `v*.*.*`.

Образы публикуются в GitLab Container Registry:

```text
$CI_REGISTRY_IMAGE/app:$CI_COMMIT_TAG
$CI_REGISTRY_IMAGE/app:latest
```

Для этого репозитория это будет формат:

```text
registry.gitlab.com/itmo-labs1/mlops/app:v1.2.3
registry.gitlab.com/itmo-labs1/mlops/app:latest
```

Release job перед сборкой копирует модель из artifacts `dvc_pipeline` в `app/models`,
затем выполняет smoke-test `/health` и `/predict`.
