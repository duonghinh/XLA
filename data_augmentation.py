import cv2
import numpy as np
import albumentations as A
import os
from tqdm import tqdm
import shutil

def create_augmentation_pipeline():
    """Tạo pipeline tăng cường dữ liệu cho biển số xe"""
    transform = A.Compose([
        # Biến đổi độ sáng và độ tương phản - phù hợp với điều kiện ánh sáng khác nhau
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        
        # Thêm nhiễu - mô phỏng camera chất lượng thấp
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        
        # Làm mờ - mô phỏng chuyển động hoặc camera không lấy nét
        A.OneOf([
            A.MotionBlur(blur_limit=7, p=0.5),
            A.MedianBlur(blur_limit=7, p=0.5),
            A.GaussianBlur(blur_limit=7, p=0.5)
        ], p=0.3),
        
        # Biến dạng hình học nhẹ - mô phỏng góc chụp khác nhau
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=10,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5
        ),
        
        # Thay đổi màu sắc
        A.OneOf([
            A.HueSaturationValue(p=0.5),
            A.RGBShift(p=0.5)
        ], p=0.3),
        
        # Thêm mưa hoặc tuyết - mô phỏng điều kiện thời tiết
        A.OneOf([
            A.RandomRain(p=0.5),
            A.RandomShadow(p=0.5)
        ], p=0.2),
    ])
    return transform

def augment_dataset(input_dir, output_dir, num_augmented=3):
    """
    Tăng cường dữ liệu cho dataset biển số xe
    
    Args:
        input_dir: Thư mục chứa ảnh gốc và file labels
        output_dir: Thư mục lưu ảnh và labels sau khi tăng cường
        num_augmented: Số ảnh tăng cường cho mỗi ảnh gốc
    """
    # Tạo thư mục output nếu chưa tồn tại
    images_output = os.path.join(output_dir, 'images')
    labels_output = os.path.join(output_dir, 'labels')
    os.makedirs(images_output, exist_ok=True)
    os.makedirs(labels_output, exist_ok=True)
    
    # Tạo transform pipeline
    transform = create_augmentation_pipeline()
    
    # Lấy danh sách ảnh từ thư mục images
    images_dir = os.path.join(input_dir, 'images')
    labels_dir = os.path.join(input_dir, 'labels')
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    print(f"Tìm thấy {len(image_files)} ảnh gốc")
    print(f"Bắt đầu tăng cường dữ liệu...")
    
    for img_file in tqdm(image_files):
        # Đọc ảnh
        image_path = os.path.join(images_dir, img_file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Copy file gốc sang thư mục output
        shutil.copy2(image_path, os.path.join(images_output, img_file))
        label_file = os.path.splitext(img_file)[0] + '.txt'
        if os.path.exists(os.path.join(labels_dir, label_file)):
            shutil.copy2(
                os.path.join(labels_dir, label_file),
                os.path.join(labels_output, label_file)
            )
        
        # Tạo các phiên bản tăng cường
        for i in range(num_augmented):
            # Áp dụng transform
            augmented = transform(image=image)
            aug_image = augmented['image']
            
            # Lưu ảnh đã tăng cường
            filename = f"{os.path.splitext(img_file)[0]}_aug_{i}.jpg"
            output_path = os.path.join(images_output, filename)
            
            # Chuyển về BGR để lưu với cv2
            aug_image = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, aug_image)
            
            # Copy label file tương ứng
            if os.path.exists(os.path.join(labels_dir, label_file)):
                aug_label = f"{os.path.splitext(img_file)[0]}_aug_{i}.txt"
                shutil.copy2(
                    os.path.join(labels_dir, label_file),
                    os.path.join(labels_output, aug_label)
                )

if __name__ == "__main__":
    # Thư mục dataset gốc (phải có cấu trúc images/ và labels/)
    input_directory = "dataset"
    # Thư mục sẽ chứa dataset đã tăng cường
    output_directory = "dataset_augmented"
    
    # Thực hiện tăng cường dữ liệu
    augment_dataset(input_directory, output_directory, num_augmented=3)
    print("Hoàn thành tăng cường dữ liệu!")