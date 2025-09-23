import matplotlib.pyplot as plt
import pandas as pd

# Đọc file results.csv
results = pd.read_csv('runs/train/exp/results.csv')

# Lấy số epoch
epochs = range(1, len(results) + 1)

# Lấy các metric
train_acc = results['metrics/precision'] * 100  # Chuyển sang phần trăm
val_acc = results['metrics/mAP_0.5'] * 100      # Chuyển sang phần trăm
train_loss = results['train/box_loss']
val_loss = results['val/box_loss']

# Tạo figure với 2 subplot
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs, train_acc, 'b-', label='tập huấn luyện')
plt.plot(epochs, val_acc, 'orange', label='tập kiểm tra/đánh giá')
plt.title('Độ chính xác theo epoch')
plt.xlabel('Epoch')
plt.ylabel('Độ chính xác (%)')
plt.legend()
plt.grid(True)
plt.ylim(50, 100)

# Plot loss 
plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, 'b-', label='tập huấn luyện')
plt.plot(epochs, val_loss, 'orange', label='tập kiểm tra')
plt.title('Hàm mất mát theo epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# In ra độ chính xác cuối cùng
print(f"Độ chính xác cuối cùng:")
print(f"Tập huấn luyện: {train_acc.iloc[-1]:.2f}%")
print(f"Tập kiểm tra: {val_acc.iloc[-1]:.2f}%")