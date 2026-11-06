FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN pip install --no-cache-dir pytest==8.3.5 coverage==7.6.10
COPY runner.py /opt/codeguard/runner.py
WORKDIR /workspace
USER 65534:65534
ENTRYPOINT ["python", "/opt/codeguard/runner.py"]
