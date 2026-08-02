FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /opt/pcpi
COPY . .
RUN python -m pip install --no-cache-dir --upgrade pip \
 && python -m pip install --no-cache-dir -e ".[dev]" \
 && python -m pytest -q \
 && python -m build
ENTRYPOINT ["pcpi-candidate-certification"]
