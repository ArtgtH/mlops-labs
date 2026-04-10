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
