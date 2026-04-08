ARG VARIANT=3.11-bookworm
FROM mcr.microsoft.com/devcontainers/python:3.11

# System dependencies for OCR and PDF processing.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        ghostscript \
        qpdf \
        unpaper \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*