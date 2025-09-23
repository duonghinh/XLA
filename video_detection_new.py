import cv2
import torch
import numpy as np
from pathlib import Path
import sys
import time

# Add YOLOv5 to path
sys.path.append('yolov5')

# Import YOLOv5 utils
from models.common import DetectMultiBackend
from utils.general import (check_img_size, non_max_suppression, scale_boxes)
from utils.torch_utils import select_device
from utils.augmentations import letterbox

def load_model(weights, device):
    model = DetectMultiBackend(weights, device=device)
    stride, names = model.stride, model.names
    return model, stride, names

def process_frame(frame, model, device, stride, names, img_size=640, is_ocr=False):
    try:
        # Preprocess frame
        img = letterbox(frame, img_size, stride=stride)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0
        if len(img.shape) == 3:
            img = img[None]

        # Inference
        with torch.no_grad():
            pred = model(img)
        
        # Different confidence thresholds for detector and OCR
        conf_thres = 0.5 if is_ocr else 0.25
        pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=0.45, max_det=1000)

        # Process detections
        detections = []
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in det:
                    detections.append((xyxy, conf, cls))
        
        return detections
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        return []

def sort_chars_by_position(chars_with_pos):
    # Sort characters by x-coordinate (left to right)
    return [char for char, _ in sorted(chars_with_pos, key=lambda x: x[1])]

def main():
    # Initialize
    device = select_device('')
    lp_detector = 'model/LP_detector.pt'
    lp_ocr = 'model/LP_ocr.pt'
    
    # Load models
    detector_model, detector_stride, detector_names = load_model(lp_detector, device)
    ocr_model, ocr_stride, ocr_names = load_model(lp_ocr, device)
    detector_size = check_img_size(640, s=detector_stride)
    ocr_size = check_img_size(640, s=ocr_stride)

    # Open video file
    video_path = 'video_test/a.mp4'
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output video writer
    output_path = 'result/output_video.mp4'
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame with detector
        detections = process_frame(frame, detector_model, device, detector_stride, detector_names, detector_size)

        # Draw detections and recognize characters
        for bbox, conf, cls in detections:
            x1, y1, x2, y2 = [int(val) for val in bbox]
            # Draw rectangle around license plate
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Crop license plate region
            plate_img = frame[y1:y2, x1:x2]
            plate_text = ""
            
            if plate_img.size > 0:
                # Process plate image with OCR model
                ocr_results = process_frame(plate_img, ocr_model, device, ocr_stride, ocr_names, ocr_size, is_ocr=True)
                
                # Get characters from OCR results with their positions
                chars_with_pos = []
                for ocr_bbox, ocr_conf, ocr_cls in ocr_results:
                    char = ocr_names[int(ocr_cls)]
                    x_pos = int(ocr_bbox[0])  # Use x-coordinate for sorting
                    chars_with_pos.append((char, x_pos))
                
                # Sort and join characters
                chars = sort_chars_by_position(chars_with_pos)
                plate_text = ''.join(chars)
            
            # Put text above the box
            label = f'Bien so: {plate_text} ({conf:.2f})'
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Write frame
        out.write(frame)
        
        # Display frame
        cv2.imshow('Nhan dien bien so xe', frame)
        
        # Break if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
