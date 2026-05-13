FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN pip install --no-cache-dir \
    matplotlib \
    mlflow \
    numpy \
    pandas \
    scikit-learn \
    kaggle \
    kagglehub \
    torch

ENV TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor

RUN mkdir -p /tmp/torchinductor

COPY . /app

CMD ["/bin/bash"]