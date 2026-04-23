# FDM 3D Print Failure Detection System

Real-time failure detection for FDM 3D printers using computer vision and edge AI. Detects spaghetti, layer shifting, and warping defects, automatically pauses the printer, and sends WhatsApp alerts.

![System Architecture](docs/architecture.png)

## Features

- **Real-time detection** — YOLOv8n model running on Raspberry Pi 4
- **90.8% mAP50** accuracy across 3 defect classes
- **2.4ms inference time** using TensorFlow Lite
- **Automatic pause** via OctoPrint API (M0 G-code)
- **Instant alerts** via WhatsApp (Twilio API)
- **Fully local** — no cloud dependencies, no subscriptions
- **Low cost** — ~$115 USD total hardware cost

## Detection Classes

| Class | Precision | Recall | mAP50 |
|-------|-----------|--------|-------|
| Spaghetti | 86.2% | 82.7% | 87.0% |
| Layer Shift | 94.1% | 95.6% | 93.8% |
| Warping | 83.4% | 87.9% | 91.6% |
| **Overall** | **87.9%** | **88.7%** | **90.8%** |

## Hardware Requirements

| Component | Specifications |
|-----------|----------------|
| Raspberry Pi 4 | 8GB RAM recommended |
| 3D Printer | Any FDM printer compatible with OctoPrint |
| Camera | IP camera or smartphone with MJPEG stream |
| MicroSD Card | 32GB+ Class 10 |

## Software Requirements

- Raspberry Pi OS (OctoPi recommended)
- Python 3.9+
- OctoPrint with API access enabled

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/fdm-failure-detection.git
cd fdm-failure-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the model

Download the TFLite model from [Google Drive]((https://drive.google.com/file/d/1HQSlUMMqAXdt0wYJzb0evgwyh2-q6hN4/view?usp=sharing)) and place it in the `models/` folder:

```
models/
└── model.tflite
```

### 4. Configure the system

Copy the example config and add your credentials:

```bash
cp config.example.py config.py
nano config.py
```

Edit `config.py` with your:
- OctoPrint API key and URL
- Twilio credentials (Account SID, Auth Token, phone numbers)
- Camera stream URL
- Detection threshold (default: 0.5)

### 5. Run the detector

```bash
python detector.py
```

## Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `CAMERA_URL` | MJPEG stream URL | — |
| `OCTOPRINT_URL` | OctoPrint server URL | `http://localhost` |
| `OCTOPRINT_API_KEY` | OctoPrint API key | — |
| `CONFIDENCE_THRESHOLD` | Minimum confidence to trigger pause | `0.5` |
| `SCAN_INTERVAL` | Seconds between scans | `5` |
| `STARTUP_DELAY` | Seconds to wait before starting detection | `60` |

## Camera Setup

Position the camera at approximately:
- **Angle:** 40° from the print bed center
- **Distance:** 25cm horizontal distance
- **Height:** 6cm above bed level
- **Orientation:** Landscape

## How It Works

1. Captures frame from IP camera MJPEG stream
2. Preprocesses image (resize to 640x640, normalize)
3. Runs YOLOv8n inference via TensorFlow Lite
4. If defect detected with confidence ≥ threshold:
   - Sends M0 pause command to OctoPrint
   - Saves detection image with bounding boxes
   - Sends WhatsApp alert via Twilio

## Project Structure

```
fdm-failure-detection/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── detector.py          # Main detection script
├── config.example.py    # Example configuration
├── models/
│   └── model.tflite     # YOLOv8n TFLite model (download separately)
├── docs/
│   └── architecture.png
└── detections/          # Saved detection images (created at runtime)
```

## Training Your Own Model

The model was trained using:
- **Dataset:** 1,231 images → 3,221 after augmentation
- **Platform:** Google Colab with Tesla T4 GPU
- **Framework:** Ultralytics YOLOv8
- **Epochs:** 100
- **Image size:** 640x640

See [Roboflow](https://roboflow.com) for dataset labeling and augmentation.

## Cost Comparison

| Item | This System | Obico Pro (3 years) |
|------|-------------|---------------------|
| Hardware | $115 | $125 |
| Subscriptions | $0 | $144 |
| **Total** | **$115** | **$269** |

## Limitations

- Reduced sensitivity for layer shifts < 5mm
- Fixed camera angle limits detection on opposite side
- Tested primarily with Ender 3 V3 SE and PLA
- Requires stable WiFi connection

## Future Work

- [ ] Second camera for warping detection
- [ ] Web-based GUI
- [ ] Multi-printer support
- [ ] Model quantization (int8)
- [ ] Community dataset expansion

## License

MIT License — see [LICENSE](LICENSE) for details.

## Author

**Sebastián Torres**  
Mechatronics Engineering — UNITEC Honduras  
[LinkedIn](https://www.linkedin.com/in/sebastiantc) · [Email](mailto:s.torrescarrasco@gmail.com)

## Acknowledgments

- Advisor: Ing. Fávell Núñez
- Universidad Tecnológica Centroamericana (UNITEC)
- [Ultralytics](https://ultralytics.com) for YOLOv8
- [Roboflow](https://roboflow.com) for dataset tools
