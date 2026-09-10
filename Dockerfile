FROM python:3.12-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py ./
COPY public ./public
COPY docs ./docs

ENV PORT=8080
ENV PUBLIC_HOST=app11.nextaura.us
ENV PUBLIC_URL=https://app11.nextaura.us
ENV TRAIN_DEVICE=cpu
ENV HEADLESS=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8080
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
