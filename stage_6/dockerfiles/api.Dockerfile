# Use an official Python image as a parent image
FROM python:3.11.9

# Install necessary dependencies (including poetry)
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get -y install curl
RUN apt-get install libgomp1

# Set the working directory in the container
WORKDIR /app

# Copy project files into the container
COPY pyproject.toml poetry.lock ./

# Upgrade pip and install poetry
RUN pip install --upgrade pip setuptools wheel
RUN pip install poetry

# Install project dependencies
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --no-root -vvv && \
    poetry run pip install --upgrade pip setuptools wheel && \
    poetry run pip install lightfm && \
    poetry cache clear pypi --all -n


# Copy the rest of the project files (including the source code)
COPY . /app

# Set the entrypoint
CMD ["poetry", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]