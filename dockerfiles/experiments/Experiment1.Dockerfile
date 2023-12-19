# Dockerfile
FROM python:3.11-slim-buster AS app

WORKDIR /app
COPY . /app

RUN apt-get update && \
    pip3 install poetry

RUN poetry config virtualenvs.create false

# Install any needed packages specified in pyproject.toml
RUN poetry install
RUN pip3 install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

CMD ["poetry", "run", "python3", "experiments/experiment1.py"]
