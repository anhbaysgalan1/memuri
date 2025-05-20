FROM python:3.11

# Set environment variables
ENV PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

# Copy requirements directly and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and project files
COPY src/ ./src/
COPY pyproject.toml .
COPY README.md .

# Make sure the module is importable
RUN pip install -e .

# Use a shell as entrypoint so we can run different commands per service
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["echo 'Specify a command to run'"] 