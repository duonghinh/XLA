import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import torch
import numpy as np
from pathlib import Path
import sys
import time
import sqlite3
from datetime import datetime
from PIL import Image, ImageTk
from collections import Counter

# Add YOLOv5 to path
sys.path.append('yolov5')

# Import YOLOv5 utils
from models.common import DetectMultiBackend
from utils.general import (check_img_size, non_max_suppression, scale_boxes)
from utils.torch_utils import select_device
from utils.augmentations import letterbox

class ParkingSystem:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Hệ Thống Quản Lý Bãi Đỗ Xe")
        self.root.geometry("1400x800")
        
        # Initialize database
        self.init_database()
        
        # Initialize models
        self.device = select_device('')
        self.detector_model, self.detector_stride, self.detector_names = self.load_model('model/LP_detector.pt')
        self.ocr_model, self.ocr_stride, self.ocr_names = self.load_model('model/LP_ocr.pt')
        self.detector_size = check_img_size(640, s=self.detector_stride)
        self.ocr_size = check_img_size(640, s=self.ocr_stride)
        
        # Initialize video variables
        self.cap = None
        self.is_capturing = False
        self.current_plate = None
        self.plate_history = []
        self.max_history = 3
        
        self.setup_gui()
        
    def load_model(self, weights):
        """Load YOLOv5 model"""
        model = DetectMultiBackend(weights, device=self.device)
        stride, names = model.stride, model.names
        model.warmup()  # Warm up model
        return model, stride, names

    def init_database(self):
        """Initialize database"""
        self.conn = sqlite3.connect('parking.db')
        self.cursor = self.conn.cursor()
        
        # Create tables if not exists
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS parking_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_number TEXT NOT NULL,
                entry_time DATETIME,
                exit_time DATETIME,
                fee REAL,
                status TEXT
            )
        ''')
        self.conn.commit()

    def setup_gui(self):
        """Setup GUI elements"""
        # Main container
        main_container = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Video and Controls
        left_panel = ttk.Frame(main_container)
        main_container.add(left_panel, weight=2)
        
        # Video frame
        self.video_frame = ttk.LabelFrame(left_panel, text="Camera View")
        self.video_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(padx=5, pady=5)
        
        # Control buttons
        control_frame = ttk.Frame(left_panel)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(control_frame, text="Chọn Video", command=self.select_video).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Camera", command=self.start_camera).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Dừng", command=self.stop_capture).pack(side=tk.LEFT, padx=5)
        
        # Right panel - Parking Records
        right_panel = ttk.Frame(main_container)
        main_container.add(right_panel, weight=1)
        
        # Current vehicles
        current_frame = ttk.LabelFrame(right_panel, text="Xe đang trong bãi")
        current_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Treeview for current vehicles
        self.current_tree = ttk.Treeview(current_frame, columns=('Biển số', 'Giờ vào'), show='headings')
        self.current_tree.heading('Biển số', text='Biển số')
        self.current_tree.heading('Giờ vào', text='Giờ vào')
        self.current_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # History frame
        history_frame = ttk.LabelFrame(right_panel, text="Lịch sử ra vào")
        history_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Treeview for history
        self.history_tree = ttk.Treeview(history_frame, 
                                       columns=('Biển số', 'Vào', 'Ra', 'Phí'),
                                       show='headings')
        self.history_tree.heading('Biển số', text='Biển số')
        self.history_tree.heading('Vào', text='Giờ vào')
        self.history_tree.heading('Ra', text='Giờ ra')
        self.history_tree.heading('Phí', text='Phí (VNĐ)')
        self.history_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Load existing records
        self.load_records()

    def select_video(self):
        """Select and process video file"""
        video_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi")])
        if video_path:
            self.stop_capture()
            self.cap = cv2.VideoCapture(video_path)
            self.is_capturing = True
            self.update_frame()

    def start_camera(self):
        """Start capturing from camera"""
        self.stop_capture()
        self.cap = cv2.VideoCapture(0)
        self.is_capturing = True
        self.update_frame()

    def stop_capture(self):
        """Stop video capture"""
        self.is_capturing = False
        if self.cap:
            self.cap.release()

    def process_frame(self, frame, model, stride, names, img_size=640, is_ocr=False):
        """Process frame with YOLO model"""
        try:
            # Preprocess frame
            img = letterbox(frame, img_size, stride=stride)[0]
            img = img.transpose((2, 0, 1))[::-1]
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.float()
            img /= 255.0
            if len(img.shape) == 3:
                img = img[None]

            # Inference
            with torch.no_grad():
                pred = model(img)
            
            # NMS
            conf_thres = 0.5 if is_ocr else 0.25
            pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=0.45)

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

    def update_frame(self):
        """Update video frame and process license plates"""
        if self.is_capturing and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Process frame for license plate detection
                detections = self.process_frame(frame, self.detector_model,
                                             self.detector_stride, self.detector_names,
                                             self.detector_size)
                
                for bbox, conf, cls in detections:
                    x1, y1, x2, y2 = [int(val) for val in bbox]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Process license plate
                    plate_img = frame[y1:y2, x1:x2]
                    if plate_img.size > 0:
                        # OCR
                        ocr_results = self.process_frame(plate_img, self.ocr_model,
                                                       self.ocr_stride, self.ocr_names,
                                                       self.ocr_size, is_ocr=True)
                        
                        # Get characters
                        chars = []
                        for ocr_bbox, ocr_conf, ocr_cls in ocr_results:
                            char = self.ocr_names[int(ocr_cls)]
                            x_pos = int(ocr_bbox[0])
                            chars.append((char, x_pos))
                        
                        # Sort and join characters
                        plate_text = ''.join(char for char, _ in sorted(chars, key=lambda x: x[1]))
                        
                        if plate_text and plate_text != self.current_plate:
                            self.handle_plate_detection(plate_text)
                            self.current_plate = plate_text
                        
                        # Draw text
                        cv2.putText(frame, f'Bien so: {plate_text}',
                                  (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                                  0.9, (0, 255, 0), 2)
                
                # Convert frame for display
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame = ImageTk.PhotoImage(frame)
                
                self.video_label.configure(image=frame)
                self.video_label.image = frame
                
                self.root.after(10, self.update_frame)

    def handle_plate_detection(self, plate_text):
        """Handle detected license plate"""
        try:
            # Check if vehicle is already in parking
            self.cursor.execute('''
                SELECT id, entry_time FROM parking_records 
                WHERE plate_number = ? AND exit_time IS NULL
                ORDER BY entry_time DESC LIMIT 1
            ''', (plate_text,))
            
            record = self.cursor.fetchone()
            current_time = datetime.now()
            
            if record:  # Vehicle is exiting
                # Calculate fee
                entry_time = datetime.strptime(record[1], '%Y-%m-%d %H:%M:%S')
                duration = (current_time - entry_time).total_seconds() / 3600  # hours
                fee = self.calculate_fee(duration)
                
                # Update record
                self.cursor.execute('''
                    UPDATE parking_records 
                    SET exit_time = ?, fee = ?, status = 'completed'
                    WHERE id = ?
                ''', (current_time.strftime('%Y-%m-%d %H:%M:%S'), fee, record[0]))
                
                messagebox.showinfo("Xe ra", 
                    f"Biển số: {plate_text}\n"
                    f"Thời gian vào: {entry_time.strftime('%H:%M:%S %d/%m/%Y')}\n"
                    f"Thời gian ra: {current_time.strftime('%H:%M:%S %d/%m/%Y')}\n"
                    f"Phí gửi xe: {fee:,.0f} VNĐ")
                    
            else:  # Vehicle is entering
                self.cursor.execute('''
                    INSERT INTO parking_records (plate_number, entry_time, status)
                    VALUES (?, ?, 'parking')
                ''', (plate_text, current_time.strftime('%Y-%m-%d %H:%M:%S')))
                
                messagebox.showinfo("Xe vào", 
                    f"Biển số: {plate_text}\n"
                    f"Thời gian: {current_time.strftime('%H:%M:%S %d/%m/%Y')}")
            
            self.conn.commit()
            self.load_records()
            
        except Exception as e:
            print(f"Error handling plate detection: {str(e)}")

    def calculate_fee(self, hours):
        """Calculate parking fee"""
        # Base fee: 5000đ/hour
        base_fee = 5000
        return round(max(base_fee, base_fee * hours))

    def load_records(self):
        """Load and display parking records"""
        # Clear current trees
        for item in self.current_tree.get_children():
            self.current_tree.delete(item)
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
            
        # Load current vehicles
        self.cursor.execute('''
            SELECT plate_number, entry_time 
            FROM parking_records 
            WHERE exit_time IS NULL 
            ORDER BY entry_time DESC
        ''')
        for record in self.cursor.fetchall():
            self.current_tree.insert('', 0, values=record)
            
        # Load history
        self.cursor.execute('''
            SELECT plate_number, entry_time, exit_time, fee 
            FROM parking_records 
            WHERE exit_time IS NOT NULL 
            ORDER BY exit_time DESC 
            LIMIT 50
        ''')
        for record in self.cursor.fetchall():
            self.history_tree.insert('', 0, values=record)

if __name__ == '__main__':
    app = ParkingSystem()
    app.root.mainloop()
