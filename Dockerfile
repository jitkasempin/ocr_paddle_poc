FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime AS base
# FROM python:3.11-slim AS base
WORKDIR /app
# RUN apt-get update && apt-get install -y git

COPY requirements.txt .
RUN apt-get update && apt-get install -y gcc && apt-get install -y poppler-utils
RUN apt-get install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender1
RUN apt-get install -y libvips-dev libvips42 tesseract-ocr
RUN python -m pip install --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu129/
RUN python -m pip install "paddleocr[all]"
RUN python -m pip install langextract
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt


FROM base AS runner

WORKDIR /app

COPY src/demo_ocr /app/src/demo_ocr

ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV PYTHONPATH "/app/src"

