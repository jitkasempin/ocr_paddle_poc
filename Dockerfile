FROM pytorch/pytorch:2.9.0-cuda13.0-cudnn9-runtime AS base

# FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime AS base
# FROM python:3.11-slim AS base
WORKDIR /app
# RUN apt-get update && apt-get install -y git

COPY requirements.txt .
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc poppler-utils \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    libvips-dev libvips42 tesseract-ocr libtesseract-dev wget \
 && rm -rf /var/lib/apt/lists/*

# Download MRZ trained data for Tesseract (passport scanning)
RUN TESSDATA_DIR=$(find /usr/share/tesseract-ocr -name "tessdata" -type d | head -1) \
 && wget -O ${TESSDATA_DIR}/mrz.traineddata \
    https://github.com/DoubangoTelecom/tesseractMRZ/raw/master/tessdata_best/mrz.traineddata

ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata/
# RUN python -m pip install paddlepaddle-gpu 
# RUN python -m pip install "paddleocr[all]"

# Python 3.12.3 virtual environment (run Streamlit from here)
RUN conda create -y -n py312 python=3.12.3 pip \
 && conda clean -afy \
 && /opt/conda/envs/py312/bin/python -m venv /opt/venv \
 && /opt/venv/bin/python -m pip install --upgrade pip setuptools wheel

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN --mount=type=cache,target=/root/.cache/pip /opt/venv/bin/pip install -r requirements.txt
RUN /opt/venv/bin/python -c "import sys; print(sys.version)"




FROM base AS runner

WORKDIR /app

COPY src/demo_ocr /app/src/demo_ocr

ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV PYTHONPATH "/app/src"

