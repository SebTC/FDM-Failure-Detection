# ===========================================
# FDM Failure Detection - Configuration
# ===========================================
# Copy this file to config.py and fill in your values
# DO NOT commit config.py to git (it contains secrets)

# --- Camera Settings ---
CAMERA_URL = "http://192.168.1.XXX:8080/video"  # Your IP camera MJPEG stream URL

# --- OctoPrint Settings ---
OCTOPRINT_URL = "http://localhost"  # Or your OctoPrint IP
OCTOPRINT_API_KEY = "YOUR_OCTOPRINT_API_KEY"  # Get from OctoPrint Settings > API

# --- Twilio Settings (for WhatsApp alerts) ---
TWILIO_ACCOUNT_SID = "YOUR_ACCOUNT_SID"
TWILIO_AUTH_TOKEN = "YOUR_AUTH_TOKEN"
TWILIO_WHATSAPP_FROM = "whatsapp:+14155238886"  # Twilio sandbox number
TWILIO_WHATSAPP_TO = "whatsapp:+1XXXXXXXXXX"  # Your phone number

# --- Detection Settings ---
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence to trigger pause (0.0 - 1.0)
SCAN_INTERVAL = 5  # Seconds between scans
STARTUP_DELAY = 60  # Seconds to wait before starting (skip purge line)

# --- Model Settings ---
MODEL_PATH = "models/model.tflite"
INPUT_SIZE = 640  # Model input size (640x640)

# --- Class Configuration ---
# Classes: 0 = layer_shift, 1 = spaghetti, 2 = warping
CLASSES = ["layer_shift", "spaghetti", "warping"]

# Classes that trigger pause (set threshold to 1.0 to disable)
CLASS_THRESHOLDS = {
    "layer_shift": 0.5,
    "spaghetti": 0.5,
    "warping": 1.0,  # Disabled - camera angle not optimal for warping
}

# --- Output Settings ---
SAVE_DETECTIONS = True
DETECTIONS_DIR = "detections"
