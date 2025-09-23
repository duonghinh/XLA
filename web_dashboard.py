from flask import Flask, render_template, jsonify, request, Response
import cv2
import torch
import sqlite3
from datetime import datetime
import threading
import base64
import numpy as np
import os
from werkzeug.utils import secure_filename

try:
    import function.helper as helper
    import function.utils_rotate as utils_rotate
    USE_ORIGINAL_HELPER = True
    print("Using original helper functions from main.py")
except:
    USE_ORIGINAL_HELPER = False
    print("Helper functions not found, using built-in")

app = Flask(__name__)

class CompleteLicensePlateSystem:
    def __init__(self):
        self.setup_database()
        self.setup_models()
        self.camera = None
        self.is_streaming = False
        
    def setup_database(self):
        self.conn = sqlite3.connect('complete_parking.db', check_same_thread=False)
        self.cursor = self.conn.cursor()
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                license_plate TEXT,
                timestamp TEXT,
                confidence REAL,
                source TEXT DEFAULT 'upload',
                image_base64 TEXT
            )
        ''')
        
        self.conn.commit()
    
    def setup_models(self):
        try:
            print("Loading AI models...")
            self.yolo_LP_detect = torch.hub.load('yolov5', 'custom', path='model/LP_detector.pt', force_reload=True, source='local')
            self.yolo_license_plate = torch.hub.load('yolov5', 'custom', path='model/LP_ocr.pt', force_reload=True, source='local')
            self.yolo_license_plate.conf = 0.60
            print("Models loaded successfully!")
            self.models_loaded = True
        except Exception as e:
            print(f"Error loading models: {e}")
            self.models_loaded = False
    
    def detect_license_plate(self, frame):
        try:
            if not self.models_loaded:
                return []
                
            plates = self.yolo_LP_detect(frame, size=640)
            list_plates = plates.pandas().xyxy[0].values.tolist()
            detected_plates = []
            list_read_plates = set()
            
            if not list_plates:
                if USE_ORIGINAL_HELPER:
                    lp = helper.read_plate(self.yolo_license_plate, frame)
                else:
                    lp = self.fallback_read_plate(frame)
                
                if lp != "unknown":
                    list_read_plates.add(lp)
                    detected_plates.append({
                        'plate': lp,
                        'confidence': 0.8,
                        'bbox': [0, 0, frame.shape[1], frame.shape[0]],
                        'timestamp': datetime.now().isoformat()
                    })
            else:
                for plate in list_plates:
                    x, y, w, h = int(plate[0]), int(plate[1]), int(plate[2] - plate[0]), int(plate[3] - plate[1])
                    confidence = plate[4]
                    crop_img = frame[y:y+h, x:x+w]
                    
                    for cc in range(2):
                        for ct in range(2):
                            if USE_ORIGINAL_HELPER:
                                lp = helper.read_plate(self.yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
                            else:
                                processed_img = self.simple_deskew(crop_img, cc, ct)
                                lp = self.fallback_read_plate(processed_img)
                            
                            if lp != "unknown":
                                list_read_plates.add(lp)
                                detected_plates.append({
                                    'plate': lp,
                                    'confidence': confidence,
                                    'bbox': [x, y, w, h],
                                    'timestamp': datetime.now().isoformat()
                                })
                                break
                        if lp != "unknown":
                            break
            
            return detected_plates
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def simple_deskew(self, img, cc, ct):
        try:
            if cc == 1:
                img = cv2.flip(img, 0)
            if ct == 1:
                rows, cols = img.shape[:2]
                M = cv2.getRotationMatrix2D((cols/2, rows/2), 2, 1)
                img = cv2.warpAffine(img, M, (cols, rows))
            return img
        except:
            return img

    def fallback_read_plate(self, img):
        try:
            results = self.yolo_license_plate(img, size=640)
            detections = results.pandas().xyxy[0]
            
            if len(detections) > 0:
                detections = detections.sort_values('xmin')
                
                plate_chars = []
                for _, detection in detections.iterrows():
                    if detection['confidence'] > 0.3:
                        class_id = int(detection['class'])
                        char = self.class_to_char_original(class_id)
                        if char:
                            x_pos = detection['xmin']
                            plate_chars.append((x_pos, char))
                
                plate_chars.sort(key=lambda x: x[0])
                plate_text = ''.join([char for _, char in plate_chars])
                
                if len(plate_text) >= 6:
                    formatted = self.format_vietnam_plate(plate_text)
                    return formatted if len(formatted) >= 4 else "unknown"
                
                return plate_text if len(plate_text) >= 4 else "unknown"
            
            return "unknown"
        except Exception as e:
            print(f"OCR Error: {e}")
            return "unknown"

    def format_vietnam_plate(self, plate_text):
        clean_text = plate_text.upper()
        
        letter_pos = -1
        for i, char in enumerate(clean_text):
            if char.isalpha():
                letter_pos = i
                break
        
        if letter_pos >= 2 and len(clean_text) >= 7:
            prefix = clean_text[:letter_pos+1]
            suffix = clean_text[letter_pos+1:]
            return f"{prefix}-{suffix}"
        
        return clean_text

    def class_to_char_original(self, class_id):
        char_map = {
            0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
            10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H',
            18: 'K', 19: 'L', 20: 'M', 21: 'N', 22: 'P', 23: 'R', 24: 'S', 25: 'T',
            26: 'U', 27: 'V', 28: 'X', 29: 'Y', 30: 'Z'
        }
        return char_map.get(class_id, '')
    
    def save_detection(self, plate_info, image=None):
        try:
            image_base64 = ""
            if image is not None:
                _, buffer = cv2.imencode('.jpg', image)
                image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            self.cursor.execute('''
                INSERT INTO detections (license_plate, timestamp, confidence, source, image_base64)
                VALUES (?, ?, ?, ?, ?)
            ''', (plate_info['plate'], plate_info['timestamp'], plate_info['confidence'], 
                  plate_info.get('source', 'upload'), image_base64))
            
            self.conn.commit()
            
        except Exception as e:
            print(f"Save error: {e}")
    
    def get_stats(self):
        try:
            conn = sqlite3.connect('complete_parking.db')
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM detections')
            total = cursor.fetchone()[0]
            
            today = datetime.now().strftime('%Y-%m-%d')
            cursor.execute('SELECT COUNT(*) FROM detections WHERE date(timestamp) = ?', (today,))
            today_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT AVG(confidence) FROM detections WHERE date(timestamp) = ?', (today,))
            avg_conf = cursor.fetchone()[0] or 0
            
            conn.close()
            
            return {
                'total_detections': total,
                'today_entries': today_count,
                'accuracy_rate': round(avg_conf * 100, 2)
            }
        except Exception as e:
            print(f"Stats error: {e}")
            return {'total_detections': 0, 'today_entries': 0, 'accuracy_rate': 0}

complete_system = CompleteLicensePlateSystem()

@app.route('/')
def dashboard():
    return '''
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Smart Parking Dashboard</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                color: #333;
            }
            .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
            .header {
                background: rgba(255,255,255,0.95);
                padding: 20px;
                border-radius: 15px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                margin-bottom: 30px;
                text-align: center;
            }
            .header h1 { color: #667eea; font-size: 2.5em; margin-bottom: 10px; }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 20px;
                margin-bottom: 30px;
            }
            .stat-card {
                background: rgba(255,255,255,0.95);
                padding: 25px;
                border-radius: 15px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                text-align: center;
            }
            .stat-value { font-size: 2.2em; font-weight: bold; color: #667eea; }
            .stat-label { color: #666; font-size: 1em; margin-top: 8px; }
            .main-content {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 30px;
                margin-bottom: 30px;
            }
            .camera-section, .upload-section {
                background: rgba(255,255,255,0.95);
                padding: 25px;
                border-radius: 15px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            }
            .camera-stream, .image-display {
                width: 100%;
                height: 300px;
                border-radius: 10px;
                background: #f0f0f0;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 15px 0;
                border: 2px dashed #ccc;
            }
            .upload-btn, .camera-btn {
                background: #667eea;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 1em;
                margin: 5px;
                transition: all 0.3s ease;
            }
            .upload-btn:hover, .camera-btn:hover { background: #5a6fd8; }
            .camera-btn.stop { background: #dc3545; }
            .camera-btn.stop:hover { background: #c82333; }
            .results-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 30px;
            }
            .results-section, .history-section {
                background: rgba(255,255,255,0.95);
                padding: 25px;
                border-radius: 15px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            }
            .result-item {
                background: #e8f5e8;
                padding: 15px;
                border-radius: 8px;
                margin: 10px 0;
                border-left: 4px solid #28a745;
            }
            .detection-item {
                background: #f8f9fa;
                padding: 12px;
                border-radius: 8px;
                margin-bottom: 10px;
                border-left: 3px solid #667eea;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .plate-text { font-weight: bold; font-size: 1.1em; color: #333; }
            .confidence { color: #28a745; font-size: 0.9em; }
            .controls {
                background: rgba(255,255,255,0.95);
                padding: 20px;
                border-radius: 15px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                text-align: center;
                margin-top: 20px;
            }
            .btn {
                background: #667eea;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 8px;
                cursor: pointer;
                margin: 5px;
            }
            .btn:hover { background: #5a6fd8; }
            .btn.danger { background: #dc3545; }
            .btn.danger:hover { background: #c82333; }
            @media (max-width: 768px) {
                .main-content, .results-grid { grid-template-columns: 1fr; }
                .stats-grid { grid-template-columns: 1fr; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Smart Parking System</h1>
                <p>Hệ thống nhận diện biển số xe thông minh</p>
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value" id="total-detections">-</div>
                    <div class="stat-label">Tổng Quét</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="today-entries">-</div>
                    <div class="stat-label">Hôm Nay</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="accuracy-rate">-</div>
                    <div class="stat-label">Độ Chính Xác (%)</div>
                </div>
            </div>
            
            <div class="main-content">
                <div class="camera-section">
                    <h2>Camera Trực Tiếp</h2>
                    <img id="camera-stream" class="camera-stream" style="display: none;">
                    <div id="camera-placeholder" class="camera-stream">
                        Nhấn "Bật Camera" để bắt đầu
                    </div>
                    <div style="text-align: center;">
                        <button class="camera-btn" id="camera-toggle" onclick="toggleCamera()">Bật Camera</button>
                    </div>
                </div>
                
                <div class="upload-section">
                    <h2>Upload Ảnh</h2>
                    <div id="image-display" class="image-display">
                        Chọn ảnh để nhận diện biển số
                    </div>
                    <div style="text-align: center;">
                        <button class="upload-btn" onclick="uploadImage()">Chọn Ảnh</button>
                        <input type="file" id="imageInput" accept="image/*" style="display: none;" onchange="processImage()">
                    </div>
                </div>
            </div>
            
            <div class="results-grid">
                <div class="results-section">
                    <h2>Kết Quả Nhận Diện</h2>
                    <div id="results-content">
                        <p style="text-align: center; color: #666;">Chưa có kết quả nào</p>
                    </div>
                </div>
                
                <div class="history-section">
                    <h2>Lịch Sử Gần Đây</h2>
                    <div id="history-content">
                        <p style="text-align: center; color: #666;">Chưa có dữ liệu</p>
                    </div>
                </div>
            </div>
            
            <div class="controls">
                <button class="btn" onclick="exportData()">Xuất Dữ Liệu</button>
                <button class="btn" onclick="viewHistory()">Lịch Sử Chi Tiết</button>
                <button class="btn danger" onclick="clearData()">Xóa Dữ Liệu</button>
            </div>
        </div>
        
        <script>
            let cameraRunning = false;
            
            setInterval(updateStats, 5000);
            setInterval(updateHistory, 10000);
            
            function updateStats() {
                fetch('/api/stats')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('total-detections').textContent = data.total_detections;
                        document.getElementById('today-entries').textContent = data.today_entries;
                        document.getElementById('accuracy-rate').textContent = data.accuracy_rate;
                    })
                    .catch(error => console.error('Stats error:', error));
            }
            
            function updateHistory() {
                fetch('/api/recent_detections')
                    .then(response => response.json())
                    .then(data => {
                        const content = document.getElementById('history-content');
                        if (data.length === 0) {
                            content.innerHTML = '<p style="text-align: center; color: #666;">Chưa có dữ liệu</p>';
                            return;
                        }
                        
                        let html = '';
                        data.forEach(item => {
                            html += `
                                <div class="detection-item">
                                    <div>
                                        <div class="plate-text">${item.plate}</div>
                                        <div style="color: #666; font-size: 0.8em;">${item.time_ago}</div>
                                    </div>
                                    <div class="confidence">${(item.confidence * 100).toFixed(1)}%</div>
                                </div>
                            `;
                        });
                        content.innerHTML = html;
                    })
                    .catch(error => console.error('History error:', error));
            }
            
            function toggleCamera() {
                const btn = document.getElementById('camera-toggle');
                const stream = document.getElementById('camera-stream');
                const placeholder = document.getElementById('camera-placeholder');
                
                if (!cameraRunning) {
                    fetch('/api/camera/start', { method: 'POST' })
                        .then(response => response.json())
                        .then(data => {
                            if (data.success) {
                                cameraRunning = true;
                                btn.textContent = 'Tắt Camera';
                                btn.className = 'camera-btn stop';
                                stream.src = '/video_feed';
                                stream.style.display = 'block';
                                placeholder.style.display = 'none';
                            } else {
                                alert('Không thể bật camera: ' + data.message);
                            }
                        });
                } else {
                    fetch('/api/camera/stop', { method: 'POST' })
                        .then(response => response.json())
                        .then(data => {
                            cameraRunning = false;
                            btn.textContent = 'Bật Camera';
                            btn.className = 'camera-btn';
                            stream.style.display = 'none';
                            placeholder.style.display = 'flex';
                        });
                }
            }
            
            function uploadImage() {
                document.getElementById('imageInput').click();
            }
            
            function processImage() {
                const input = document.getElementById('imageInput');
                const file = input.files[0];
                
                if (file) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        document.getElementById('image-display').innerHTML = 
                            `<img src="${e.target.result}" style="max-width: 100%; max-height: 100%; object-fit: contain;">`;
                    };
                    reader.readAsDataURL(file);
                    
                    const formData = new FormData();
                    formData.append('image', file);
                    
                    document.getElementById('results-content').innerHTML = '<p>Đang xử lý...</p>';
                    
                    fetch('/api/process_image', {
                        method: 'POST',
                        body: formData
                    })
                    .then(response => response.json())
                    .then(data => {
                        let html = '';
                        if (data.success && data.detections && data.detections.length > 0) {
                            data.detections.forEach(detection => {
                                html += `
                                    <div class="result-item">
                                        <strong>Biển số: ${detection.plate}</strong><br>
                                        <span style="color: #28a745;">Độ tin cậy: ${(detection.confidence * 100).toFixed(1)}%</span>
                                    </div>
                                `;
                            });
                        } else {
                            html = '<p style="color: #dc3545;">Không phát hiện biển số nào!</p>';
                        }
                        
                        document.getElementById('results-content').innerHTML = html;
                        updateStats();
                        updateHistory();
                    })
                    .catch(error => {
                        console.error('Process error:', error);
                        document.getElementById('results-content').innerHTML = 
                            '<p style="color: #dc3545;">Lỗi xử lý ảnh!</p>';
                    });
                }
            }
            
            function exportData() {
                window.open('/api/export', '_blank');
            }
            
            function viewHistory() {
                window.open('/history', '_blank');
            }
            
            function clearData() {
                if (confirm('Bạn có chắc muốn xóa toàn bộ dữ liệu?')) {
                    fetch('/api/clear_data', { method: 'POST' })
                        .then(response => response.json())
                        .then(data => {
                            alert(data.message);
                            updateStats();
                            updateHistory();
                        });
                }
            }
            
            updateStats();
            updateHistory();
        </script>
    </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    def generate():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return
            
        while complete_system.is_streaming:
            ret, frame = cap.read()
            if not ret:
                break
            
            detections = complete_system.detect_license_plate(frame)
            
            for detection in detections:
                x, y, w, h = detection['bbox']
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{detection['plate']} ({detection['confidence']:.2f})", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                detection['source'] = 'camera'
                complete_system.save_detection(detection, frame)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        cap.release()
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/camera/start', methods=['POST'])
def start_camera():
    try:
        complete_system.is_streaming = True
        return jsonify({'success': True, 'message': 'Camera đã bật!'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Lỗi: {str(e)}'})

@app.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    complete_system.is_streaming = False
    return jsonify({'success': True, 'message': 'Camera đã tắt!'})

@app.route('/api/stats')
def get_stats():
    return jsonify(complete_system.get_stats())

@app.route('/api/recent_detections')
def get_recent_detections():
    try:
        complete_system.cursor.execute('''
            SELECT license_plate, timestamp, confidence FROM detections 
            ORDER BY timestamp DESC LIMIT 10
        ''')
        
        detections = []
        for row in complete_system.cursor.fetchall():
            plate, timestamp, confidence = row
            try:
                dt = datetime.fromisoformat(timestamp)
                time_diff = datetime.now() - dt
                
                if time_diff.seconds < 60:
                    time_ago = f"{time_diff.seconds} giây trước"
                elif time_diff.seconds < 3600:
                    time_ago = f"{time_diff.seconds // 60} phút trước"
                else:
                    time_ago = f"{time_diff.seconds // 3600} giờ trước"
            except:
                time_ago = "Vừa xong"
            
            detections.append({
                'plate': plate,
                'confidence': confidence,
                'time_ago': time_ago
            })
        
        return jsonify(detections)
    except Exception as e:
        print(f"Recent detections error: {e}")
        return jsonify([])

@app.route('/api/process_image', methods=['POST'])
def process_uploaded_image():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'Không có file ảnh'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Không chọn file'})
        
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'success': False, 'error': 'Không thể đọc ảnh'})
        
        detections = complete_system.detect_license_plate(image)
        
        for detection in detections:
            detection['source'] = 'upload'
            complete_system.save_detection(detection, image)
        
        return jsonify({
            'success': True,
            'detections': detections,
            'message': f'Phát hiện {len(detections)} biển số'
        })
        
    except Exception as e:
        print(f"Process image error: {e}")
        return jsonify({'success': False, 'error': f'Lỗi xử lý: {str(e)}'})

@app.route('/api/export')
def export_data():
    try:
        complete_system.cursor.execute('SELECT * FROM detections ORDER BY timestamp DESC')
        data = complete_system.cursor.fetchall()
        
        csv_content = "ID,License Plate,Timestamp,Confidence,Source\n"
        for row in data:
            csv_content += f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]}\n"
        
        response = Response(csv_content, mimetype='text/csv')
        response.headers['Content-Disposition'] = f'attachment; filename=license_plates_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        return response
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history')
def history_page():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Lịch Sử Nhận Diện</title>
        <meta charset="UTF-8">
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 0; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container { 
                max-width: 1200px; 
                margin: 0 auto; 
                background: rgba(255,255,255,0.95); 
                padding: 30px; 
                border-radius: 15px; 
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            }
            h1 { color: #667eea; text-align: center; margin-bottom: 30px; }
            .search-bar {
                margin-bottom: 20px;
                display: flex;
                gap: 10px;
                align-items: center;
                justify-content: center;
            }
            .search-bar input {
                padding: 10px;
                border: 2px solid #667eea;
                border-radius: 8px;
                font-size: 16px;
                width: 300px;
            }
            .search-bar button {
                padding: 10px 20px;
                background: #667eea;
                color: white;
                border: none;
                border-radius: 8px;
                cursor: pointer;
            }
            .search-bar button:hover { background: #5a6fd8; }
            table { 
                width: 100%; 
                border-collapse: collapse; 
                margin-top: 20px; 
                background: white;
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            th, td { padding: 15px; text-align: left; border-bottom: 1px solid #eee; }
            th { 
                background: #667eea; 
                color: white; 
                font-weight: bold;
                text-transform: uppercase;
                font-size: 0.9em;
                letter-spacing: 0.5px;
            }
            tr:hover { background-color: #f8f9fa; }
            .plate { 
                font-weight: bold; 
                color: #333; 
                background: #e8f5e8;
                padding: 5px 10px;
                border-radius: 15px;
                display: inline-block;
                }
            .confidence { 
                color: #28a745; 
                font-weight: bold;
            }
            .source {
                padding: 3px 8px;
                border-radius: 12px;
                font-size: 0.8em;
                text-transform: uppercase;
                font-weight: bold;
            }
            .source.upload { background: #e3f2fd; color: #1976d2; }
            .source.camera { background: #f3e5f5; color: #7b1fa2; }
            .timestamp {
                color: #666;
                font-size: 0.9em;
            }
            .loading {
                text-align: center;
                padding: 50px;
                color: #666;
                font-size: 1.2em;
            }
            .no-data {
                text-align: center;
                padding: 50px;
                color: #999;
                font-style: italic;
            }
            .stats-row {
                background: #f8f9fa;
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 20px;
                display: flex;
                justify-content: space-around;
                text-align: center;
            }
            .stat-item {
                display: flex;
                flex-direction: column;
            }
            .stat-number {
                font-size: 2em;
                font-weight: bold;
                color: #667eea;
            }
            .stat-label {
                color: #666;
                font-size: 0.9em;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Lịch Sử Nhận Diện Biển Số</h1>
            
            <div class="stats-row" id="stats-row">
                <div class="stat-item">
                    <div class="stat-number" id="total-count">-</div>
                    <div class="stat-label">Tổng số</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number" id="today-count">-</div>
                    <div class="stat-label">Hôm nay</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number" id="avg-confidence">-</div>
                    <div class="stat-label">Độ tin cậy TB</div>
                </div>
            </div>
            
            <div class="search-bar">
                <input type="text" id="searchInput" placeholder="Tìm kiếm biển số..." onkeyup="searchPlates()">
                <button onclick="searchPlates()">Tìm kiếm</button>
                <button onclick="loadHistory()">Làm mới</button>
                <button onclick="window.close()">Đóng</button>
            </div>
            
            <table id="history-table">
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Biển Số</th>
                        <th>Thời Gian</th>
                        <th>Độ Tin Cậy</th>
                        <th>Nguồn</th>
                    </tr>
                </thead>
                <tbody id="history-body">
                    <tr><td colspan="5" class="loading">Đang tải dữ liệu...</td></tr>
                </tbody>
            </table>
        </div>
        
        <script>
            let allData = [];
            
            function loadHistory() {
                document.getElementById('history-body').innerHTML = 
                    '<tr><td colspan="5" class="loading">Đang tải dữ liệu...</td></tr>';
                
                fetch('/api/all_detections')
                    .then(response => response.json())
                    .then(data => {
                        allData = data;
                        displayData(data);
                        updateStats(data);
                    })
                    .catch(error => {
                        console.error('Error:', error);
                        document.getElementById('history-body').innerHTML = 
                            '<tr><td colspan="5" class="no-data">Lỗi tải dữ liệu</td></tr>';
                    });
            }
            
            function displayData(data) {
                const tbody = document.getElementById('history-body');
                
                if (data.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="5" class="no-data">Chưa có dữ liệu nào</td></tr>';
                    return;
                }
                
                let html = '';
                data.forEach(item => {
                    const date = new Date(item.timestamp);
                    const formattedDate = date.toLocaleString('vi-VN');
                    
                    html += `
                        <tr>
                            <td>${item.id}</td>
                            <td><span class="plate">${item.plate}</span></td>
                            <td class="timestamp">${formattedDate}</td>
                            <td class="confidence">${(item.confidence * 100).toFixed(1)}%</td>
                            <td><span class="source ${item.source}">${item.source}</span></td>
                        </tr>
                    `;
                });
                tbody.innerHTML = html;
            }
            
            function updateStats(data) {
                const total = data.length;
                const today = new Date().toDateString();
                const todayCount = data.filter(item => 
                    new Date(item.timestamp).toDateString() === today
                ).length;
                
                const avgConfidence = total > 0 ? 
                    (data.reduce((sum, item) => sum + item.confidence, 0) / total * 100).toFixed(1) + '%' : '0%';
                
                document.getElementById('total-count').textContent = total;
                document.getElementById('today-count').textContent = todayCount;
                document.getElementById('avg-confidence').textContent = avgConfidence;
            }
            
            function searchPlates() {
                const searchTerm = document.getElementById('searchInput').value.toLowerCase();
                
                if (searchTerm === '') {
                    displayData(allData);
                    return;
                }
                
                const filteredData = allData.filter(item => 
                    item.plate.toLowerCase().includes(searchTerm)
                );
                
                displayData(filteredData);
            }
            
            loadHistory();
        </script>
    </body>
    </html>
    '''

@app.route('/api/all_detections')
def get_all_detections():
    try:
        complete_system.cursor.execute('''
            SELECT id, license_plate, timestamp, confidence, source FROM detections 
            ORDER BY timestamp DESC LIMIT 1000
        ''')
        
        detections = []
        for row in complete_system.cursor.fetchall():
            detections.append({
                'id': row[0],
                'plate': row[1],
                'timestamp': row[2],
                'confidence': row[3],
                'source': row[4] if len(row) > 4 else 'upload'
            })
        
        return jsonify(detections)
    except Exception as e:
        print(f"All detections error: {e}")
        return jsonify([])

@app.route('/api/clear_data', methods=['POST'])
def clear_data():
    try:
        complete_system.cursor.execute('DELETE FROM detections')
        complete_system.conn.commit()
        return jsonify({'message': 'Đã xóa toàn bộ dữ liệu!', 'status': 'success'})
    except Exception as e:
        return jsonify({'message': f'Lỗi: {str(e)}', 'status': 'error'})

if __name__ == '__main__':
    print("Starting Complete Parking Dashboard...")
    print("Dashboard: http://localhost:5000")
    print("Camera Feed: http://localhost:5000/video_feed") 
    print("History: http://localhost:5000/history")
    print("Features:")
    print("  - Camera streaming with real-time detection")
    print("  - Image upload with result display")
    print("  - Live statistics")
    print("  - Detailed history with search")
    print("  - Data export to CSV")
    print("\nPress Ctrl+C to stop")
    
    app.run(host='127.0.0.1', port=5000, debug=True, threaded=True)