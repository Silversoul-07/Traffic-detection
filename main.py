import torch
import cv2
import os
import numpy as np
from ultralytics import YOLO

class TrafficDetection:
    def __init__(self, model_path="models/yolo11x.pt", confidence_threshold=0.5):
        self.model = YOLO(model_path)
        self.model.to(torch.device("cuda:0"))
        self.model.eval()
        self.classes = self.model.names
        
        # Add input size configuration
        self.input_width = 640
        self.input_height = 320
        self.model.imgsz = (self.input_height, self.input_width)  # Set model input size
        
        # Increase confidence threshold for better classification
        self.conf_threshold = confidence_threshold
        
        # Reference dimensions in meters for different objects
        self.reference_dimensions = {
            0: 1.7,    # person height
            2: 4.5,    # car length
            3: 2.0,    # motorcycle length
            5: 12.0,   # bus length
            7: 15.0    # truck length
        }
        
        # Calibration parameters
        self.focal_length = None
        self.calibration_distance = 10
        self.calibration_height = None

        # Add counter dictionary
        self.class_counters = {}

        # Add class filtering to handle misclassifications
        self.class_mapping = self.model.names.copy()
        
        # Debug mode
        self.debug_mode = False

    def detect_objects(self, frame):
        # Preserve aspect ratio during resize
        h, w = frame.shape[:2]
        scale = min(self.input_width/w, self.input_height/h)
        new_w, new_h = int(w*scale), int(h*scale)
        
        # Resize frame while preserving aspect ratio
        resized = cv2.resize(frame, (new_w, new_h))
        
        # Create black canvas of target size
        canvas = np.zeros((self.input_height, self.input_width, 3), dtype=np.uint8)
        
        # Place the resized image in the center of the canvas
        x_offset = (self.input_width - new_w) // 2
        y_offset = (self.input_height - new_h) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        # Run YOLOv8 inference with tracking
        results = self.model.track(canvas, persist=True, conf=self.conf_threshold)[0]
        detections = []
        
        if self.debug_mode:
            if hasattr(results.boxes, 'cls'):
                class_counts = {}
                for cls in results.boxes.cls:
                    cls_id = int(cls.item())
                    cls_name = self.classes[cls_id]
                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                print(f"Classes detected: {class_counts}")
        
        if results.boxes.id is not None:
            boxes = results.boxes.data.tolist()
            track_ids = results.boxes.id.tolist()
            
            for box, track_id in zip(boxes, track_ids):
                # The box format appears to be [x1, y1, x2, y2, unknown_value, conf, cls]
                x1, y1, x2, y2, _, conf, cls = box if len(box) >= 7 else box[:6] + [0]
                
                # Apply class mapping if exists
                cls_int = int(cls)
                
                w = x2 - x1
                h = y2 - y1
                x = x1
                y = y1
                
                # Adjust coordinates from padded image back to original scale
                if x_offset > 0 or y_offset > 0:
                    x = max(0, x - x_offset) / scale
                    y = max(0, y - y_offset) / scale
                    w = w / scale
                    h = h / scale
                
                detections.append({
                    'box': [int(x), int(y), int(w), int(h)], 
                    'confidence': conf, 
                    'class_id': cls_int, 
                    'track_id': int(track_id)
                })
        
        return detections

    def update_counters(self, detections):
        """Update class counters"""
        # Reset counters
        self.class_counters = {}
        
        # Count objects by class
        for det in detections:
            class_id = det['class_id']
            if class_id in self.classes:
                class_name = self.classes[class_id]
                if class_name in self.class_counters:
                    self.class_counters[class_name] += 1
                else:
                    self.class_counters[class_name] = 1

    def display_counters(self, frame):
        """Display object counts in top-right corner"""
        if not self.class_counters:
            return frame
        
        # Set display parameters
        padding = 10
        line_height = 30
        font_scale = 0.7
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255)  #c:\Users\praveen\AppData\Local\Packages\Microsoft.ScreenSketch_8wekyb3d8bbwe\TempState\Recordings\20250410-0718-26.5068294.mp4 White text
        
        # Sort class names for consistent display
        class_names = sorted(self.class_counters.keys())
        
        # Calculate board dimensions
        n_classes = len(class_names)
        board_height = n_classes * line_height + 2 * padding
        
        # Get the width needed for text (include some extra space)
        text_samples = [f"{cls}: {self.class_counters[cls]}" for cls in class_names]
        max_text_width = 0
        for text in text_samples:
            text_size = cv2.getTextSize(text, font, font_scale, 2)[0][0]
            max_text_width = max(max_text_width, text_size)
        
        board_width = max_text_width + 2 * padding + 20  # Add extra padding
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        board_x = frame.shape[1] - board_width  # Right side
        board_y = 0  # Top
        cv2.rectangle(overlay, (board_x, board_y), 
                     (frame.shape[1], board_height), 
                     (0, 0, 0), -1)  # Black background
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)  # Add transparency
        
        # Draw class names with counts
        y = padding + line_height
        for cls in class_names:
            text = f"{cls}: {self.class_counters[cls]}"
            cv2.putText(frame, text, 
                      (board_x + padding, y), 
                      font, font_scale, color, 2)
            y += line_height
            
        return frame

    def estimate_distance(self, obj_height_pixels, class_id):
        """Estimate distance using object height and known dimensions"""
        if class_id not in self.reference_dimensions:
            return None
            
        # Auto-calibrate focal length using first detection
        if self.focal_length is None and obj_height_pixels > 0:
            real_height = self.reference_dimensions[class_id]
            self.focal_length = (obj_height_pixels * self.calibration_distance) / real_height
            
        if self.focal_length and obj_height_pixels > 0:
            real_height = self.reference_dimensions[class_id]
            distance = (real_height * self.focal_length) / obj_height_pixels
            return distance
        return None

    def process_frame(self, frame):
        # Resize input frame
        frame = cv2.resize(frame, (self.input_width, self.input_height))
        result_frame = frame.copy()
        
        # Detect objects
        detections = self.detect_objects(frame)
        
        # Update counters
        self.update_counters(detections)
        
        # Draw detections
        for detection in detections:
            x, y, w, h = detection['box']
            class_id = detection['class_id']
            track_id = detection['track_id']
            
            # Estimate distance
            distance = self.estimate_distance(h, class_id)
            
            # Draw bounding box
            color = (0, 255, 0)  # Green for all objects
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, 2)
            
            label = f"ID:{track_id} {self.classes[class_id]}"
            if distance:
                label += f" {distance:.1f}m"
            
            # Put label above the bounding box
            cv2.putText(result_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Add warning for close objects
            if distance and distance < 5:
                cv2.putText(result_frame, "WARNING: Object too close!", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display object counters in top-right corner
        result_frame = self.display_counters(result_frame)
        
        return result_frame

def main():
    os.mkdir("models", exist_ok=True)
    default_model = "models/yolo11x.pt"  # Or use your custom model
    
    detector = TrafficDetection(model_path=default_model, confidence_threshold=0.6)
    
    # Open video file
    video_path = "samples/tc1.mp4"
    cap = cv2.VideoCapture(video_path)
    
    # Set fixed window size
    window_name = 'Traffic Detection'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 320)
    
    # Process video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        result = detector.process_frame(frame)
        cv2.imshow(window_name, result)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord('f'):  # Toggle fullscreen
            current_prop = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
            new_prop = cv2.WINDOW_NORMAL if current_prop == cv2.WINDOW_FULLSCREEN else cv2.WINDOW_FULLSCREEN
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, new_prop)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()