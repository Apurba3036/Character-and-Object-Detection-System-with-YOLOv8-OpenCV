# 🖼️ Character and Object Detection System with YOLOv8 & OpenCV

A comprehensive detection system built using YOLOv8 and OpenCV, capable of detecting and tracking both characters and objects in images and videos with high accuracy and real-time performance.

## 🎯 Example Results


### 🎥 Video Detection
<img width="565" height="386" alt="image" src="https://github.com/user-attachments/assets/6cbb1162-284a-4fba-a75b-841a6117c7e3" />


Bounding boxes with character and object labels are displayed in real-time.
## 🚀 Features

- 🧠 YOLOv8 Model for state-of-the-art detection
- 👥 Character detection and recognition
- 🎯 Multiple object class detection
- 🎥 Real-time Video Processing using OpenCV
- ⚡ Lightweight & Fast inference

## 🛠️ Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/Object-and-Character-Detection-YOLOv8.git
    cd Object-and-Character-Detection-YOLOv8
    ```

2. Create a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux / Mac
    venv\Scripts\activate     # Windows
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 📦 Dependencies

- Python 3.8+
- OpenCV
- Ultralytics YOLOv8
- NumPy

Install them manually if needed:
```bash
pip install opencv-python ultralytics numpy
```

## 📥 Model Files

The following model files need to be downloaded separately due to size limitations:
- YOLOv8x model: Download from [Ultralytics](https://github.com/ultralytics/yolov8/releases)
- CRAFT text detection model: Download from [CRAFT-pytorch](https://drive.google.com/file/d/1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ/view)

Place the downloaded files in the `models/` directory.

## ▶️ Usage

### 🔹 Run on an image
```bash
python detect.py --image images/sample.jpg
```

### 🔹 Run on a video
```bash
python detect.py --video videos/sample.mp4
```

### 🔹 Run on webcam
```bash
python detect.py --webcam
```

## 📊 Model Performance

| Model Variant | Speed (ms) | mAP@50 | Parameters |
|--------------|------------|--------|------------|
| YOLOv8n      | 1.2        | 37.3   | 3.2M       |
| YOLOv8s      | 1.9        | 44.9   | 11.2M      |
| YOLOv8m      | 2.8        | 50.2   | 25.9M      |

## 🏗️ Future Improvements

- ✅ Character recognition enhancement
- ✅ Multi-character tracking
- ✅ Custom character dataset training
- ✅ Deploy as a web application




