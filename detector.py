"""
FDM 3D Print Failure Detection System
Real-time detection of spaghetti, layer shift, and warping defects.

Author: Sebastián Torres
License: MIT
"""

import requests
import time
import numpy as np
from io import BytesIO
from PIL import Image, ImageDraw
import tflite_runtime.interpreter as tflite
import datetime
import os

# Import configuration
try:
    from config import *
except ImportError:
    print("ERROR: config.py not found!")
    print("Copy config.example.py to config.py and fill in your credentials.")
    exit(1)

# === LOAD MODEL ===
print("Loading model...")
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
print(f"Model loaded! Input shape: {input_shape}")


def preprocess_image(img):
    """Resize and normalize image for model"""
    img = img.resize((input_shape[2], input_shape[1]))
    img = np.array(img, dtype=np.float32)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def get_snapshot():
    """Capture frame from IP camera MJPEG stream"""
    try:
        response = requests.get(CAMERA_URL, timeout=5, stream=True)
        bytes_data = b''
        for chunk in response.iter_content(chunk_size=1024):
            bytes_data += chunk
            start = bytes_data.find(b'\xff\xd8')
            end = bytes_data.find(b'\xff\xd9')
            if start != -1 and end != -1:
                jpg = bytes_data[start:end+2]
                img = Image.open(BytesIO(jpg)).convert('RGB')
                img = img.rotate(-90, expand=True)
                return img
    except Exception as e:
        print(f"Camera error: {e}")
        return None


def pause_print():
    """Pause the print by sending M0 command via OctoPrint"""
    try:
        headers = {"X-Api-Key": OCTOPRINT_API_KEY, "Content-Type": "application/json"}
        data = {"command": "M0"}
        response = requests.post(
            f"{OCTOPRINT_URL}/api/printer/command",
            json=data,
            headers=headers
        )
        print(">>> PRINT PAUSED!")
        return response.ok
    except Exception as e:
        print(f"Pause error: {e}")
        return False


def resume_print():
    """Resume the print by sending M108 command via OctoPrint"""
    try:
        headers = {"X-Api-Key": OCTOPRINT_API_KEY, "Content-Type": "application/json"}
        data = {"command": "M108"}
        requests.post(f"{OCTOPRINT_URL}/api/printer/command", json=data, headers=headers)
        print(">>> PRINT RESUMED!")
    except Exception as e:
        print(f"Resume error: {e}")


def detect(img):
    """Run detection on image"""
    input_data = preprocess_image(img)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data


def parse_detections(output):
    """Parse YOLOv8 output for detections above threshold"""
    detections = []
    predictions = output[0].T  # Shape (8400, 7)
    
    # Debug: show max scores
    class_scores_all = predictions[:, 4:7]
    print(f"Max class scores: {np.max(class_scores_all, axis=0)}")
    
    for pred in predictions:
        class_scores = pred[4:7]
        max_score = np.max(class_scores)
        class_id = np.argmax(class_scores)
        class_name = CLASSES[class_id]
        
        # Use class-specific threshold
        threshold = CLASS_THRESHOLDS.get(class_name, CONFIDENCE_THRESHOLD)
        
        if max_score >= threshold:
            detections.append({
                'class': class_name,
                'confidence': float(max_score)
            })
    
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)[:5]
    return detections


def send_whatsapp_alert(message):
    """Send WhatsApp notification via Twilio"""
    try:
        from twilio.rest import Client
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        client.messages.create(
            body=message,
            from_=TWILIO_WHATSAPP_FROM,
            to=TWILIO_WHATSAPP_TO
        )
        print(">>> WhatsApp sent!")
    except Exception as e:
        print(f"WhatsApp error: {e}")


def log_detection(message):
    """Log detection to file and console"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = os.path.join(DETECTIONS_DIR, "detection_log.txt")
    with open(log_path, "a") as f:
        f.write(f"{timestamp} - {message}\n")
    print(message)


def draw_detections(img, detections, output):
    """Draw bounding boxes on image"""
    draw = ImageDraw.Draw(img)
    
    img_w, img_h = img.size
    scale_x = img_w / INPUT_SIZE
    scale_y = img_h / INPUT_SIZE
    
    predictions = output[0].T
    for pred in predictions:
        class_scores = pred[4:7]
        max_score = np.max(class_scores)
        class_id = np.argmax(class_scores)
        class_name = CLASSES[class_id]
        threshold = CLASS_THRESHOLDS.get(class_name, CONFIDENCE_THRESHOLD)
        
        if max_score >= threshold:
            cx, cy, w, h = pred[0:4]
            x1 = int((cx - w/2) * scale_x)
            y1 = int((cy - h/2) * scale_y)
            x2 = int((cx + w/2) * scale_x)
            y2 = int((cy + h/2) * scale_y)
            
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1-15), f"{class_name} {max_score:.0%}", fill="red")
    
    return img


def run_detection():
    """Main detection loop"""
    # Create detections directory if it doesn't exist
    os.makedirs(DETECTIONS_DIR, exist_ok=True)
    
    print("=" * 50)
    print("FDM Failure Detection System Active")
    print("=" * 50)
    print(f"Camera URL: {CAMERA_URL}")
    print(f"OctoPrint URL: {OCTOPRINT_URL}")
    print(f"Scan interval: {SCAN_INTERVAL}s")
    print(f"Startup delay: {STARTUP_DELAY}s")
    print("=" * 50)
    
    print(f"Waiting {STARTUP_DELAY} seconds for print to start...")
    time.sleep(STARTUP_DELAY)
    print("Now monitoring...")
    
    while True:
        img = get_snapshot()
        
        if img:
            output = detect(img)
            detections = parse_detections(output)
            
            if detections:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Save image with bounding boxes
                if SAVE_DETECTIONS:
                    img_with_boxes = draw_detections(img.copy(), detections, output)
                    save_path = os.path.join(DETECTIONS_DIR, f"failure_{timestamp}.jpg")
                    img_with_boxes.save(save_path)
                
                for det in detections:
                    log_detection(f"DETECTED: {det['class'].upper()} ({det['confidence']:.1%})")
                
                # Pause print
                print("PAUSING PRINT...")
                pause_print()
                
                # Send WhatsApp alert
                alert_time = datetime.datetime.now().strftime("%H:%M:%S")
                top_detection = detections[0]
                send_whatsapp_alert(
                    f"🚨 [{alert_time}] Print paused!\n"
                    f"Detected: {top_detection['class']} ({top_detection['confidence']:.0%})"
                )
                
                # Wait for user input to resume
                print("=" * 50)
                print("Detection PAUSED.")
                print("Fix the issue, then press Enter to resume...")
                input()
                
                resume_print()
                
                print("Detection RESUMED.")
                print("=" * 50)
            else:
                print("Normal")
        
        time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    run_detection()
