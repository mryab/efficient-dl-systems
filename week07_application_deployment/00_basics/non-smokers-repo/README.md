## UV workflow

```bash
uv venv .venv
source .venv/bin/activate
uv lock
uv sync --frozen --all-groups
```

- `pyproject.toml` declares acceptable ranges.
- `uv.lock` pins exact versions actually installed.

## Run locally

```bash
source .venv/bin/activate
uv sync --frozen --all-groups
uv run uvicorn app.main:app --host 0.0.0.0 --port 8080
```

## Run tests

```bash
uv run pytest -m unit
uv run pytest -m integration
```

## Docker run

```bash
docker build -f Dockerfile.good -t non-smoker-demo .
docker run --rm -p 8080:8080 non-smoker-demo
```

## Compose run

```bash
docker compose up --build
```

If you re-enable Redis blocks, send a few `/predict` requests and verify the counter:

```bash
docker compose exec redis redis-cli GET app_http_inference_count
```

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"you are an idiot"}'
```
