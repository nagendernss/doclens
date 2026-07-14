FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY doclens ./doclens
COPY web ./web
COPY evals ./evals

RUN pip install --no-cache-dir -e .[web]

EXPOSE 10000

CMD ["sh", "-c", "uvicorn doclens.server:app --host 0.0.0.0 --port ${PORT:-10000}"]
