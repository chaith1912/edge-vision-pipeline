import os
import time
import json
import cv2
import numpy as np
import redis
import torch
from ultralytics import YOLO

# 1. Hardware Verification Link
print(f"[Inference] CUDA Execution Engine Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[Inference] Targeted GPU Driver Core: {torch.cuda.get_device_name(0)}")

# 2. Establish Communication Paths
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
db = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)

# 3. Mount and Load YOLO Engine Model Weights
print("[Inference] Preparing neural layers...")
# Using yolov8n.pt triggers the hardcoded framework auto-download mirrors cleanly inside Docker
WEIGHTS_PATH = "yolov8n.pt"

print(f"[Inference] Initializing weights from {WEIGHTS_PATH}...")
model = YOLO(WEIGHTS_PATH)

def run_inference():
    print("[Inference] Subscribing to Redis frame queue pipeline...")
    
    # We poll the latest element inside the stream
    last_id = '$'
    
    while True:
        try:
            # Read from the camera:frames stream buffer
            response = db.xread({"camera:frames": last_id}, count=1, block=1000)
            if not response:
                continue
                
            stream_name, messages = response[0]
            for message_id, data in messages:
                last_id = message_id
                
                # A. Extract structural bytes and timestamp metrics
                frame_id = data[b'frame_id'].decode('utf-8')
                image_bytes = data[b'image']
                start_time = time.time()

                # B. Decode raw JPEGs back into numerical matrices
                np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                # C. Compute Detections via GPU Passthrough Acceleration
                # verbose=False keeps the console clean, device=0 forces the script onto the GPU
                results = model(frame, verbose=False, device=0)
                
                # D. Parse Spatial Vectors
                detections = []
                for box in results[0].boxes:
                    xyxy = box.xyxy[0].tolist() # Bounding box edges [x_min, y_min, x_max, y_max]
                    conf = float(box.conf[0])   # Accuracy certainty score
                    cls = int(box.cls[0])       # Object classification index (e.g., 0 for person)
                    
                    detections.append({
                        "box": [int(v) for v in xyxy],
                        "confidence": round(conf, 2),
                        "class": model.names[cls]
                    })

                # E. Calculate Inference Pipeline Latency
                latency = (time.time() - start_time) * 1000 # Milliseconds latency

                # F. Publish Coordinate Array back to Redis Vector Stream
                payload = {
                    "frame_id": frame_id,
                    "latency_ms": round(latency, 2),
                    "detections": json.dumps(detections)
                }
                
                db.xadd("camera:predictions", payload, maxlen=100, approximate=True)

        except redis.ConnectionError:
            print("[Inference] Connection to Broker dropped. Re-establishing link...")
            time.sleep(2)
        except Exception as e:
            print(f"[Inference] Operational Warning: {str(e)}")
            time.sleep(0.1)

if __name__ == "__main__":
    run_inference()