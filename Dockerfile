FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace/ct-mcot

COPY pyproject.toml README.md requirements.txt ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY docs ./docs

RUN pip install --no-cache-dir -e .

CMD ["python", "scripts/run_experiment.py", "--config", "configs/base.yaml"]
