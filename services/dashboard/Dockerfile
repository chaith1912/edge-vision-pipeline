FROM python:3.10-slim

# Install system GUI framework runtime dependencies for headless OpenCV transformations
RUN apt-get update && apt-get install -y --no-install-recommends \
    glib-2.0 \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

COPY services/dashboard/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY services/dashboard/src/ ./src/

EXPOSE 5000

CMD ["python", "src/main.py"]