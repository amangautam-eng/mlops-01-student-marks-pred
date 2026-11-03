FROM python:3.8-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install Python dependencies separately so Docker can reuse this layer when
# only application code changes.
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY application.py ./
COPY src ./src
COPY templates ./templates
COPY artifacts ./artifacts
COPY setup.py ./

EXPOSE 5000

CMD ["python", "application.py"]
