# FormFlow (Linux). Build from the repository root:
#   docker build -f docker/Dockerfile -t formflow .                      # CUDA builds (default)
#   docker build -f docker/Dockerfile --build-arg PROFILE=cpu -t formflow .
# Run (the workspace volume keeps models and sessions between containers):
#   docker run --gpus all -p 8780:8780 -v formflow-ws:/workspace formflow
# The PyTorch CUDA wheel bundles CUDA + cuDNN, which ONNX Runtime reuses: the host only needs the NVIDIA
# driver and the NVIDIA Container Toolkit. Browsers allow the camera on http://localhost; for other
# machines add `--https` to the command (self-signed certificate) or put the app behind a TLS proxy.
FROM python:3.12-slim

ARG PROFILE=gpu
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 FORMFLOW_HOME=/workspace
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements ./requirements
RUN pip install -r requirements/torch-${PROFILE}.txt && pip install -r requirements/onnxruntime-${PROFILE}.txt
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs
COPY samples ./samples
COPY benchmarks ./benchmarks
COPY docker/entrypoint.sh /entrypoint.sh
RUN pip install -e . && chmod +x /entrypoint.sh \
    && useradd -m -u 1000 ff && mkdir -p /workspace && chown ff /workspace
USER ff
VOLUME /workspace
EXPOSE 8780
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8780/api/health')"
ENTRYPOINT ["/entrypoint.sh"]
CMD ["serve", "--host", "0.0.0.0", "--no-browser"]
