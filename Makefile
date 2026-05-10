DOCKER_COMPOSE := docker compose
DOCKER_COMPOSE_RUN := $(DOCKER_COMPOSE) run --rm

build:
	$(DOCKER_COMPOSE) up --build -d app

compose-bash:
	$(DOCKER_COMPOSE_RUN) app bash

test:
	$(DOCKER_COMPOSE_RUN) app poetry run pytest -q

lint-fix:
	$(DOCKER_COMPOSE_RUN) app poetry run ruff check . --fix

format:
	$(DOCKER_COMPOSE_RUN) app poetry run ruff format .

dvc-repro:
	cd app && POETRY_VIRTUALENVS_CREATE=false poetry install --only main,pipeline --no-root
	MLFLOW_TRACKING_URI=$${MLFLOW_TRACKING_URI:-file:mlruns} dvc repro

api-image:
	mkdir -p app/models
	cp models/model.pkl models/feature_columns.json app/models/
	docker build --target deploy_build -t fraud-api app
