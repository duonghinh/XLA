import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import torch
import numpy as np
from pathlib import Path
import sys
import time
import sqlite3
from datetime import datetime
from PIL import Image, ImageTk
import json
from collections import deque

# Add YOLOv5 to path
sys.path.append('yolov5')

# Import YOLOv5 utils
from models.common import DetectMultiBackend
from utils.general import (check_img_size, non_max_suppression, scale_boxes)
from utils.torch_utils import select_device
from utils.augmentations import letterbox

# Cấu hình tối ưu cho nhận dạng biển số
CONFIG = {
    'detector': {
        'conf_threshold': 0.45,  # Ngưỡng tin cậy cho detector
        'iou_threshold': 0.45,   # Ngưỡng IOU cho NMS
        'img_size': 640,        # Kích thước ảnh đầu vào
        'max_det': 10,          # Số lượng detection tối đa
    },
    'ocr': {
        'conf_threshold': 0.6,   # Ngưỡng tin cậy cho OCR
        'iou_threshold': 0.3,    # Ngưỡng IOU cho NMS
        'img_size': 320,        # Kích thước ảnh OCR
        'max_det': 20,          # Số ký tự tối đa
    },
    'image_processing': {
        'clahe_clip_limit': 3.0,
        'clahe_grid_size': (8, 8),
        'blur_kernel_size': (3, 3),
        'sharpen_kernel': np.array([[-1,-1,-1],
                                  [-1, 9,-1],
                                  [-1,-1,-1]]),
        'threshold_block_size': 19,
        'threshold_C': 9
    },
    'stabilization': {
        'history_size': 10,     # Số frame lịch sử
        'min_detections': 3,    # Số lần xuất hiện tối thiểu
        'confidence_boost': 1.1  # Hệ số tăng độ tin cậy cho ký tự ổn định
    }
}
