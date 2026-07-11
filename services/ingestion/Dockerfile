# Step 1: Use a lightweight official Python baseline engine image
FROM python:3.11-slim

# Step 2: Install basic OS dependencies required by OpenCV image processing libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*
    
# Step 3: Define the working folder path inside our virtual container
WORKDIR /usr/src/app

# Step 4: Copy our dependency manifest and install them inside the container environment
COPY services/ingestion/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy our raw code engine directory structure over into the container workspace
COPY services/ingestion/src/ ./src/

# Step 6: Define the execution command when the container initializes
CMD ["python", "-u", "src/main.py"]