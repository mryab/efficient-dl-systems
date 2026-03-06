## Local run

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8080
```

## Docker run

```bash
docker build -f Dockerfile.bad -t smoker-demo .
docker run --rm -p 8080:8080 smoker-demo
```

## Endpoint

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"you are an idiot"}'
```
