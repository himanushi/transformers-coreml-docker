FROM python:3.8-slim-buster

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install transformers coremltools torch

WORKDIR /app

COPY convert.py .

CMD [ "python", "convert.py" ]
