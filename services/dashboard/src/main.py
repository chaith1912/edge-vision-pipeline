import os
import json
import cv2
import numpy as np
import redis
from flask import Flask, Response, render_template_string

app = Flask(__name__)

# 1. Establish Memory Broker Pipelines
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
db = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)

# 2. Minimalist UI Layout Document
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Aegis Edge Vision Control Room</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 20px; text-align: center; }
        .container { max-width: 900px; margin: 0 auto; }
        h1 { color: #38bdf8; font-size: 24px; margin-bottom: 5px; }
        .meta { color: #94a3b8; font-size: 14px; margin-bottom: 20px; }
        .monitor-frame { border: 4px solid #1e293b; border-radius: 8px; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.5); max-width: 100%; background: #000; }
        .badge { background: #0369a1; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>AEGIS EDGE VISION CONTROL PANEL</h1>
        <p class="meta">Status: <span class="badge">LIVE PLATFORM PIPELINE</span> | Channel: broker:6379 -> RTX 4050 GPU Acceleration Core</p>
        <img class="monitor-frame" src="/video_feed" alt="Live Spatial Tracking Feed">
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

def generate_stream():
    """Continuously extracts frame matrices and matches vector inferences to draw the final UI stream."""
    last_frame_id = '$'
    
    while True:
        try:
            # A. Fetch the latest raw frame array from the Redis buffer
            frame_data = db.xread({"camera:frames": last_frame_id}, count=1, block=1000)
            if not frame_data:
                continue
                
            messages = frame_data[0][1]
            for msg_id, data in messages:
                last_frame_id = msg_id
                frame_id = data[b'frame_id'].decode('utf-8')
                image_bytes = data[b'image']
                
                # B. Decode raw compressed array into pixel arrays
                np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                # C. Check for accompanying vector predictions matching this frame
                pred_data = db.xread({"camera:predictions": "$"}, count=1, block=5)
                latency_val = "N/A"
                
                # Fallback scan to match records if predictions lag or burst ahead
                latest_pred = db.xrevrange("camera:predictions", max="+", min="-", count=1)
                
                if latest_pred:
                    _, p_payload = latest_pred[0]
                    pred_frame_id = p_payload[b'frame_id'].decode('utf-8')
                    
                    # Overlay coordinates if telemetry belongs to the active processing context
                    if pred_frame_id == frame_id:
                        latency_val = f"{p_payload[b'latency_ms'].decode('utf-8')} ms"
                        detections = json.loads(p_payload[b'detections'].decode('utf-8'))
                        
                        for det in detections:
                            box = det["box"]
                            conf = det["confidence"]
                            cls_name = det["class"]
                            
                            # Draw High-Contrast Target Rectangles (Sky Blue)
                            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (248, 189, 56), 2)
                            
                            # Embed Class Label Flags
                            label = f"{cls_name.upper()} {conf}"
                            cv2.putText(img, label, (box[0], max(box[1] - 8, 15)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (248, 189, 56), 2)

                # D. Render Telemetry HUD Bar directly on Canvas Top Edge
                cv2.rectangle(img, (0, 0), (320, 35), (30, 23, 15), -1)
                hud_text = f"COMPUTE LATENCY: {latency_val}"
                cv2.putText(img, hud_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
                # E. Re-encode the local tracking canvas matrix into an HTTP MJPEG Boundary frame
                ret, buffer = cv2.imencode('.jpg', img)
                frame_out = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_out + b'\r\n')
                       
        except Exception as e:
            time.sleep(0.03)

@app.route('/video_feed')
def video_feed():
    """Exposes the MJPEG stream channel to the browser image element."""
    return Response(generate_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)