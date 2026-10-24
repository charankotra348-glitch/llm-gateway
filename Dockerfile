# Production image: the gateway installed as a package, run as a non-root user.
# Migrations run on start, then uvicorn serves with one worker: circuit breakers, rate-limit
# buckets and the response cache live in that process.

FROM python:3.12-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /src
COPY pyproject.toml ./
COPY gateway ./gateway
RUN pip install --prefix=/install .

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PORT=8000 \
    DATABASE_URL=sqlite+aiosqlite:////data/gateway.db
RUN useradd --create-home --uid 10001 app && mkdir /data && chown app /data
COPY --from=build /install /usr/local
WORKDIR /app
COPY alembic.ini routes.yaml prices.yaml ./
COPY migrations ./migrations
COPY deploy ./deploy
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/healthz', timeout=2)"
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn gateway.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*' --no-access-log"]
