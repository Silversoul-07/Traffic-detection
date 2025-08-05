# Traffic Detection System

A real-time traffic detection and analysis system using YOLOv11 for object detection and tracking. This system can detect, track, and count various traffic objects including vehicles, pedestrians, and estimate their distances from the camera.

## Features

- **Real-time Object Detection**: Detects multiple traffic objects (cars, trucks, buses, motorcycles, persons)
- **Object Tracking**: Maintains consistent tracking IDs across frames
- **Distance Estimation**: Estimates object distance using camera calibration
- **Object Counting**: Real-time counting of detected objects by class
- **Warning System**: Alerts when objects are too close (< 5 meters)
- **Optimized Performance**: GPU acceleration support with CUDA
- **Flexible Input**: Supports video files and live camera feeds

## Requirements

### System Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended)
- OpenCV-compatible camera or video files

### Dependencies

```bash
pip install torch torchvision
pip install ultralytics
pip install opencv-python
pip install numpy
```

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/traffic-detection.git
cd traffic-detection
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download YOLOv11 model** (automatically downloaded on first run)
```bash
# The model will be automatically downloaded to models/yolo11x.pt
# Or you can manually download and place it in the models/ directory
```

4. **Prepare your video files**
```bash
mkdir samples
# Place your video files in the samples/ directory
```

## Usage

### Basic Usage

```bash
python traffic_detection.py
```

### Configuration Options

The `TrafficDetection` class accepts several parameters for customization:

```python
detector = TrafficDetection(
    model_path="models/yolo11x.pt",  # Path to YOLO model
    confidence_threshold=0.6         # Detection confidence threshold
)
```

### Keyboard Controls

While the application is running:
- **'q'**: Quit the application
- **'f'**: Toggle fullscreen mode

## Project Structure

```
traffic-detection/
├── traffic_detection.py      # Main application file
├── models/                   # YOLO model files
│   └── yolo11x.pt           # YOLOv11 model (auto-downloaded)
├── samples/                  # Video samples directory
│   └── tc1.mp4              # Sample traffic video
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Technical Details

### Object Detection Classes

The system can detect and track the following object classes:
- **Person** (Class ID: 0) - Reference height: 1.7m
- **Car** (Class ID: 2) - Reference length: 4.5m  
- **Motorcycle** (Class ID: 3) - Reference length: 2.0m
- **Bus** (Class ID: 5) - Reference length: 12.0m
- **Truck** (Class ID: 7) - Reference length: 15.0m

### Distance Estimation

The system uses camera calibration and known object dimensions to estimate distances:
1. **Auto-calibration**: Automatically calibrates using the first detected object
2. **Reference dimensions**: Uses real-world object sizes for accurate distance calculation
3. **Focal length estimation**: Dynamically calculates camera focal length

### Performance Optimizations

- **GPU acceleration**: Automatically uses CUDA if available
- **Aspect ratio preservation**: Maintains image quality during processing
- **Optimized input size**: Uses 640x320 resolution for balanced speed/accuracy
- **Efficient tracking**: Maintains object IDs across frames

## Configuration

### Input Resolution
```python
self.input_width = 640
self.input_height = 320
```

### Confidence Threshold
```python
self.conf_threshold = 0.6  # Adjust for detection sensitivity
```

### Distance Warning Threshold
```python
if distance and distance < 5:  # Warning for objects closer than 5 meters
```

## Customization

### Adding New Object Classes

To add support for new object classes:

1. Add reference dimensions in `__init__`:
```python
self.reference_dimensions = {
    0: 1.7,    # person height
    2: 4.5,    # car length
    # Add new class: reference_dimension
    8: 3.0,    # example: bicycle length
}
```

2. The system will automatically detect and count any objects that YOLO can identify.

### Custom Model

To use a custom trained model:

```python
detector = TrafficDetection(model_path="path/to/your/custom_model.pt")
```

## Acknowledgments

- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics) for the object detection model
- [OpenCV](https://opencv.org/) for computer vision operations
- [PyTorch](https://pytorch.org/) for deep learning framework
