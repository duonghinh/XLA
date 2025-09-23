from PIL import Image
import cv2
import torch
import function.utils_rotate as utils_rotate
import os
import function.helper as helper
import tkinter as tk
from tkinter import filedialog, messagebox

# Load mô hình YOLO
yolo_LP_detect = torch.hub.load('yolov5', 'custom', path='model/LP_detector.pt', force_reload=True, source='local')
yolo_license_plate = torch.hub.load('yolov5', 'custom', path='model/LP_ocr.pt', force_reload=True, source='local')
yolo_license_plate.conf = 0.60

def process_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        messagebox.showerror("Lỗi", "Không thể đọc file ảnh! Kiểm tra lại đường dẫn.")
        return
    detect_and_display(img)

def process_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Lỗi", "Không thể mở webcam!")
        return
    print("🎥 Webcam đang chạy... Nhấn 'q' để thoát")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        detect_and_display_realtime(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def detect_and_display_realtime(img):
    """Phiên bản realtime cho webcam"""
    plates = yolo_LP_detect(img, size=640)
    list_plates = plates.pandas().xyxy[0].values.tolist()
    list_read_plates = set()
    
    if not list_plates:
        lp = helper.read_plate(yolo_license_plate, img)
        if lp != "unknown":
            cv2.putText(img, lp, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            list_read_plates.add(lp)
    else:
        for plate in list_plates:
            x, y, w, h = int(plate[0]), int(plate[1]), int(plate[2] - plate[0]), int(plate[3] - plate[1])
            crop_img = img[y:y+h, x:x+w]
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
            for cc in range(2):
                for ct in range(2):
                    lp = helper.read_plate(yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
                    if lp != "unknown":
                        list_read_plates.add(lp)
                        cv2.putText(img, lp, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                        break
                if lp != "unknown":
                    break
    
    cv2.imshow('License Plate Recognition - Press Q to quit', img)

def detect_and_display(img):
    """Phiên bản cho ảnh tĩnh"""
    plates = yolo_LP_detect(img, size=640)
    list_plates = plates.pandas().xyxy[0].values.tolist()
    list_read_plates = set()
    
    if not list_plates:
        lp = helper.read_plate(yolo_license_plate, img)
        if lp != "unknown":
            cv2.putText(img, lp, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            list_read_plates.add(lp)
    else:
        for plate in list_plates:
            x, y, w, h = int(plate[0]), int(plate[1]), int(plate[2] - plate[0]), int(plate[3] - plate[1])
            crop_img = img[y:y+h, x:x+w]
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
            for cc in range(2):
                for ct in range(2):
                    lp = helper.read_plate(yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
                    if lp != "unknown":
                        list_read_plates.add(lp)
                        cv2.putText(img, lp, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                        break
                if lp != "unknown":
                    break
    
    cv2.imshow('License Plate Recognition - Press any key to close', img)
    print("📋 Biển số phát hiện:", list_read_plates if list_read_plates else "Không phát hiện")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    print("🚀 Khởi động chương trình nhận diện biển số...")
    
    root = tk.Tk()
    root.title("Nhận diện biển số xe")
    
    # Đưa cửa sổ lên trên và focus
    root.lift()
    root.attributes('-topmost', True)
    root.after_idle(root.attributes, '-topmost', False)
    
    # Tạo cửa sổ nhỏ thay vì ẩn
    root.geometry("300x100")
    root.resizable(False, False)
    
    # Tạo label hướng dẫn
    label = tk.Label(root, text="Chọn chế độ nhận diện biển số:", font=("Arial", 12))
    label.pack(pady=10)
    
    # Tạo frame cho buttons
    button_frame = tk.Frame(root)
    button_frame.pack(pady=10)
    
    def choose_webcam():
        root.destroy()
        process_webcam()
    
    def choose_image():
        root.destroy()
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh chứa biển số", 
            filetypes=[("Image files", "*.jpg *.png *.jpeg *.bmp *.tiff")]
        )
        if file_path:
            process_image(file_path)
        else:
            print("❌ Không có file ảnh nào được chọn!")
    
    def exit_program():
        root.destroy()
        print("👋 Thoát chương trình!")
    
    # Tạo các nút
    btn_webcam = tk.Button(button_frame, text="📹 Webcam", command=choose_webcam, width=10)
    btn_image = tk.Button(button_frame, text="🖼️ Ảnh", command=choose_image, width=10)
    btn_exit = tk.Button(button_frame, text="❌ Thoát", command=exit_program, width=10)
    
    btn_webcam.pack(side=tk.LEFT, padx=5)
    btn_image.pack(side=tk.LEFT, padx=5)
    btn_exit.pack(side=tk.LEFT, padx=5)
    
    # Hiển thị cửa sổ
    root.mainloop()

if __name__ == "__main__":
    main()