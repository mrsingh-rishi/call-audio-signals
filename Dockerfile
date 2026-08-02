# Single image: API + static UI + ffmpeg. One deploy, one URL.
#
# No torch, no node. The T0 probe showed semantic customer identification works
# on a summed mono mix, which removed the need for a local diarization stack -
# and with it roughly 2 GB of model weights. The image is small enough for a
# 512 MB instance, so the deployment does not need a 2 GB tier.
FROM python:3.12-slim

# ffmpeg does all decoding, downmixing and resampling. curl is used once, at
# build time, to fetch the segmentation model.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Dependencies first so source edits do not invalidate the layer.
COPY pyproject.toml README.md ./
COPY src/autoace/__init__.py src/autoace/__init__.py
RUN pip install --no-cache-dir ".[server,gemini,overlap]"

# Path D's checkpoint: pyannote/segmentation-3.0 (MIT, Copyright (c) 2022 CNRS),
# ONNX export maintained by k2-fsa. Baked in at BUILD time, never fetched at
# boot - a cold container must not block on a network round trip, and the
# deployment has to keep working if the release URL ever moves.
ARG SEG_MODEL_URL=https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
RUN mkdir -p models \
 && curl -fsSL "$SEG_MODEL_URL" -o /tmp/seg.tar.bz2 \
 && tar xjf /tmp/seg.tar.bz2 -C models \
 && rm /tmp/seg.tar.bz2 \
 # Keep only what inference needs; the archive also ships example scripts.
 && find models -type f ! -name 'model.onnx' ! -name 'LICENSE' -delete \
 && test -f models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx

COPY src/ src/
COPY server/ server/
COPY web/ web/

# Non-root, with a writable location for staged uploads and the SQLite file.
RUN useradd --create-home --uid 10001 app \
 && mkdir -p /app/uploads \
 && chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health',timeout=4).status==200 else 1)"

# Single worker: the workload is I/O-bound on the Gemini API and concurrency is
# handled inside the process by an asyncio semaphore. Extra workers would
# multiply memory for no throughput gain and would split the SQLite writer.
CMD ["sh", "-c", "uvicorn main:app --app-dir server --host 0.0.0.0 --port ${PORT:-8000}"]
