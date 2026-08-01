# syntax=docker/dockerfile:1.7

ARG CUDA_IMAGE_VERSION
ARG UBUNTU_VERSION

FROM nvidia/cuda:${CUDA_IMAGE_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS base

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CASCADE_HOME=/opt/cascade \
    CASCADE_VENV=/opt/cascade/venv

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-dev python3-pip python3-venv \
      build-essential binutils ninja-build git ca-certificates libgomp1 tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/cascade


FROM base AS builder

COPY requirements.lock pyproject.toml README.md ./
COPY docker/versions.env docker/versions.env
RUN python3 -m venv "${CASCADE_VENV}" \
    && set -a \
    && . docker/versions.env \
    && set +a \
    && "${CASCADE_VENV}/bin/python" -m pip install --upgrade \
      "pip==${PIP_VERSION}" \
      "setuptools==${SETUPTOOLS_VERSION}" \
      "wheel==${WHEEL_VERSION}" \
    && "${CASCADE_VENV}/bin/python" -m pip install -r requirements.lock

COPY . /opt/cascade
RUN chmod +x docker/scripts/*.sh docker/entrypoint.sh docker/healthcheck.sh \
    && docker/scripts/install-core.sh


FROM builder AS runtime-builder

ARG PROVIDER_BUNDLE=generic
ARG IMAGE_TYPE=runtime
ARG GIT_COMMIT=unverified
RUN docker/scripts/install-provider-bundle.sh "${PROVIDER_BUNDLE}" \
    && "${CASCADE_VENV}/bin/python" docker/scripts/generate-image-manifest.py \
      --root /opt/cascade \
      --bundle "${PROVIDER_BUNDLE}" \
      --image-type "${IMAGE_TYPE}" \
      --git-commit "${GIT_COMMIT}" \
    && find /opt/cascade/layer_streaming/providers \
      -path '*/_build/*.so' -type f -delete \
    && docker/scripts/validate-image.sh build


ARG CUDA_IMAGE_VERSION
ARG UBUNTU_VERSION
FROM nvidia/cuda:${CUDA_IMAGE_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime

ARG DEBIAN_FRONTEND=noninteractive
ARG CASCADE_UID=1000
ARG CASCADE_GID=1000
ARG CASCADE_VERSION
ARG PYTORCH_VERSION
ARG CUDA_VERSION
ARG PROVIDER_ABI
ARG EXECUTION_PLAN_SCHEMA
ARG RUN_REPORT_SCHEMA
LABEL org.opencontainers.image.title="Cascade-LLM" \
      org.opencontainers.image.version="${CASCADE_VERSION}" \
      io.cascade-llm.torch.version="${PYTORCH_VERSION}" \
      io.cascade-llm.cuda.version="${CUDA_VERSION}" \
      io.cascade-llm.provider.abi="${PROVIDER_ABI}" \
      io.cascade-llm.execution-plan.schema="${EXECUTION_PLAN_SCHEMA}" \
      io.cascade-llm.run-report.schema="${RUN_REPORT_SCHEMA}"
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CASCADE_HOME=/opt/cascade \
    CASCADE_VENV=/opt/cascade/venv \
    CASCADE_MODEL_ROOT=/models \
    CASCADE_CONFIG_ROOT=/config \
    CASCADE_CACHE_ROOT=/cache/cascade \
    CASCADE_RESULT_ROOT=/results \
    HF_HOME=/cache/huggingface \
    PATH=/opt/cascade/venv/bin:${PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 ca-certificates libgomp1 tini \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid "${CASCADE_GID}" cascade \
    && useradd --uid "${CASCADE_UID}" --gid "${CASCADE_GID}" \
      --create-home --shell /bin/bash cascade \
    && mkdir -p /models /config /cache/huggingface /cache/cascade /results \
      /opt/cascade/providers /opt/cascade/adapters \
    && chown -R cascade:cascade /cache /results

COPY --from=runtime-builder /opt/cascade /opt/cascade
COPY docker/entrypoint.sh /usr/local/bin/cascade-entrypoint
COPY docker/healthcheck.sh /usr/local/bin/cascade-healthcheck
RUN chmod 0555 /usr/local/bin/cascade-entrypoint \
    /usr/local/bin/cascade-healthcheck \
    && chmod -R a-w /opt/cascade \
    && chmod -R u+rwX /cache /results

WORKDIR /opt/cascade
USER cascade
ENTRYPOINT ["/usr/bin/tini", "--", "cascade-entrypoint"]
CMD ["doctor"]
HEALTHCHECK --interval=30s --timeout=15s --start-period=20s --retries=3 \
  CMD ["cascade-healthcheck"]


FROM builder AS devel

ENV PATH=/opt/cascade/venv/bin:${PATH} \
    CASCADE_MODEL_ROOT=/models \
    CASCADE_CONFIG_ROOT=/config \
    CASCADE_CACHE_ROOT=/cache/cascade \
    CASCADE_RESULT_ROOT=/results \
    HF_HOME=/cache/huggingface
RUN mkdir -p /models /config /cache/huggingface /cache/cascade /results \
    && "${CASCADE_VENV}/bin/python" docker/scripts/generate-image-manifest.py \
      --root /opt/cascade --bundle generic --image-type devel
ENTRYPOINT ["/usr/bin/tini", "--", "/opt/cascade/docker/entrypoint.sh"]
CMD ["shell"]
