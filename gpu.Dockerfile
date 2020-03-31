FROM horovod/horovod:0.19.1-tf2.1.0-torch1.4.0-mxnet1.6.0-py3.6-gpu

ARG POETRY_VERSION=1.0.2
ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1

USER root

# Install Poetry
RUN pip install --upgrade pip
RUN pip install poetry==${POETRY_VERSION}

# Install Dependencies & Project
WORKDIR /app

COPY pyproject.toml poetry.lock /app/
RUN poetry config virtualenvs.create false
RUN poetry install

COPY . /app
RUN poetry install

# Download model
RUN poetry run download_spacy_model
