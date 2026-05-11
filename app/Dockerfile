ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS base
ENV \
  PIP_NO_CACHE_DIR=1 \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  POETRY_HOME=/opt/poetry \
  POETRY_VIRTUALENVS_CREATE=false \
  POETRY_NO_INTERACTION=1 \
  POETRY_VERSION=2.3.4 \
  POETRY_NO_ANSI=1 \
  PATH="/opt/poetry/bin:${PATH}" \
  PYTHONPATH=/app/src

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl  \
    build-essential \
    make \
    tree \
  && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python - --version ${POETRY_VERSION}

WORKDIR /app
COPY ./poetry.lock ./pyproject.toml /app/

FROM base AS dev_build

RUN poetry install --all-groups --no-root

COPY . .

FROM base AS test_build

RUN poetry install --only main,test --no-root

COPY . .

FROM base AS deploy_builder

RUN poetry install --only main --no-root

COPY ./src /app/src
COPY ./main.py /app/main.py
COPY ./models /app/models

FROM python:${PYTHON_VERSION}-slim AS deploy_build
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src
WORKDIR /app

COPY --from=deploy_builder /usr/local /usr/local
COPY --from=deploy_builder /app/src /app/src
COPY --from=deploy_builder /app/main.py /app/main.py
COPY --from=deploy_builder /app/models /app/models

RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app
USER appuser

CMD ["granian", "--interface", "asgi", "main:app", "--host", "0.0.0.0", "--port", "8000"]
