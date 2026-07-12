# Step 1: Use a production runtime base containing optimized CUDA system paths
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Step 2: Set basic environment variables to bypass interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Step 3: Build fundamental OS and system runtime configurations
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    glib-2.0 \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Step 4: Map python alias to ensure terminal executes cleanly
RUN ln -s /usr/bin/python3 /usr/bin/python

# Step 5: Define the internal workspace directory environment
WORKDIR /usr/src/app

# Step 6: Install PyTorch and application modules directly inside the container context
COPY services/inference/requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Step 7: Transfer deep learning system logic code structures
COPY services/inference/src/ ./src/

# Step 8: Initialize computation logic execution automatically on startup
CMD ["python", "-u", "src/main.py"]