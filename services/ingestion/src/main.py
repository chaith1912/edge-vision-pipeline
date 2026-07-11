import os
import time
import cv2
import redis

# 1. Connect to our Redis Broker container using environment variables passed by Docker
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

print(f"[Ingestion] Connecting to Redis at {REDIS_HOST}:{REDIS_PORT}...")
db = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)

# 2. Setup video source. 
# We use a standard public sample video URL so it works seamlessly inside the container immediately.
VIDEO_SOURCE = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/bolt-detection.mp4"

def start_ingestion():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    if not cap.isOpened():
        print("[Ingestion] Error: Could not open video source stream.")
        return

    print("[Ingestion] Video channel successfully initialized. Starting stream...")
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        
        # If the sample video ends, loop it back to the beginning automatically
        if not ret:
            print("[Ingestion] End of video stream. Restarting video cycle...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # A. Downscale image frame to 640x640 to keep performance ultra-fast for the AI model
        resized_frame = cv2.resize(frame, (640, 640))

        # B. Compress raw frame matrix into lightweight JPEG format bytes
        _, buffer = cv2.imencode('.jpg', resized_frame)
        frame_bytes = buffer.tobytes()

        # C. Push to Redis Stream using a strict Max Length constraint (maxlen=100)
        # This acts as our shock absorber ring buffer: it keeps the latest 100 frames and drops old ones
        # so your computer's system memory never fills up.
        try:
            db.xadd(
                "camera:frames", 
                {"frame_id": str(frame_count), "image": frame_bytes}, 
                maxlen=100, 
                approximate=True
            )
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"[Ingestion] Successfully streamed {frame_count} frames into Redis buffer.")
                
        except redis.ConnectionError:
            print("[Ingestion] Redis Connection lost. Retrying in 2 seconds...")
            time.sleep(2)
            continue

        # Mimic standard ~30 Frames Per Second velocity
        time.sleep(0.033)

    cap.release()

if __name__ == "__main__":
    start_ingestion()