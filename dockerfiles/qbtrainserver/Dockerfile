# qbtrainserver image — the `qbtrain` library + Django server with ALL heavy
# dependencies (and the Chromium browser for Playwright) baked in, so deploy
# targets only `docker pull` + run and never build on the VM.
#
# Build context = the qbtrain-be repo root:
#   docker build -f dockerfiles/qbtrainserver/Dockerfile -t pavanreddysec/qbtrainserver:latest .

FROM python:3.11

# Releases are tag-only: the version MUST be supplied from the git tag. There is
# no default and no fallback — building without it fails immediately.
ARG QBTRAIN_VERSION
RUN test -n "${QBTRAIN_VERSION}" \
    || { echo "ERROR: build-arg QBTRAIN_VERSION is required (releases are tag-only; tag vX.Y.Z and let CI build)."; exit 1; }

LABEL org.opencontainers.image.title="qbtrainserver" \
      org.opencontainers.image.version="${QBTRAIN_VERSION}" \
      org.opencontainers.image.source="https://github.com/pavanreddyml/qbtrain-be"

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    SETUPTOOLS_SCM_PRETEND_VERSION=${QBTRAIN_VERSION}

WORKDIR /usr/src/app

# Base OS tools (certs / fonts / timezone). Chromium's own system libraries are
# installed by `playwright install --with-deps` below.
RUN apt-get update && apt-get install -y --no-install-recommends \
      wget ca-certificates fonts-liberation tzdata \
    && rm -rf /var/lib/apt/lists/*

# 1) Library + all its (heavy) Python deps FIRST. This layer is cached and only
#    rebuilds when the library or its dependencies change.
COPY python/qbtrain ./python/qbtrain
RUN pip install ./python/qbtrain

# 2) Chromium browser + its OS libraries (playwright is a qbtrain dependency).
RUN playwright install --with-deps chromium

# 3) Server code LAST — editing it doesn't invalidate the dependency layers.
COPY python/qbtrainserver ./python/qbtrainserver

EXPOSE 8000
CMD ["python", "python/qbtrainserver/manage.py", "runserver", "0.0.0.0:8000"]
