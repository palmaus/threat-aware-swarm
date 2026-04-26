FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_CACHE_DIR=/app/.cache/pip \
    PYTHONPATH=/app \
    SDL_VIDEODRIVER=dummy \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/app/.cache/matplotlib \
    NUMBA_CACHE_DIR=/app/.cache/numba \
    TORCH_HOME=/app/.cache/torch \
    XDG_CACHE_HOME=/app/.cache

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    swig \
    libgl1 \
    libglib2.0-0 \
    graphviz \
    libsdl2-2.0-0 \
    libsdl2-image-2.0-0 \
    libsdl2-mixer-2.0-0 \
    libsdl2-ttf-2.0-0 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt constraints.txt requirements.lock /app/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt -c requirements.lock

COPY . /app

ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} app && useradd -m -u ${UID} -g ${GID} app
RUN mkdir -p /app/.cache && chown -R app:app /app/.cache
USER app

CMD ["python", "-m", "scripts.train.trained_ppo"]
