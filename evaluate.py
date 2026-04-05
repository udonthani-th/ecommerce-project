import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# --- จุดที่ 1: แก้ไขการ Import ให้ดึงจากโฟลเดอร์ src ---
from src import data_setup, model_builder 
from torchvision import transforms

def evaluate():
    # 1. ตั้งค่าพื้นฐาน
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 2. เตรียม Transformation (ต้องเหมือนกับตอน Train)
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. โหลด DataLoader (ใช้ Test Set เท่านั้น)
    print("[INFO] กำลังโหลด Test Data...")
    _, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir="data/train",
        test_dir="data/test",
        transform=data_transform,
        batch_size=32
    )

    # 4. โหลดโมเดลที่เทรนเสร็จแล้ว
    print("[INFO] กำลังโหลดโมเดลจาก models/ecommerce_model.pth...")
    model = model_builder.create_model(num_classes=len(class_names)).to(DEVICE)
    model.load_state_dict(torch.load("models/ecommerce_model.pth", map_location=DEVICE))
    model.eval()

    # 5. เริ่มทำ Prediction
    all_preds = []
    all_labels = []

    print("[INFO] เริ่มการทำ Prediction บน Test Set...")
    with torch.inference_mode():
        for X, y in test_dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            outputs = model(X)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # 6. สร้างและแสดงผล Confusion Matrix
    print("[INFO] กำลังสร้าง Confusion Matrix...")
    cm = confusion_matrix(all_labels, all_preds)
    
    # วาดรูป
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", ax=ax, xticks_rotation=45)
    
    plt.title("Confusion Matrix: E-commerce Classification")
    plt.tight_layout()
    
    # --- จุดที่ 2: บันทึกรูปภาพลงในโฟลเดอร์ results ---
    target_dir = "results"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    save_path = os.path.join(target_dir, "confusion_matrix.png")
    plt.savefig(save_path)
    print(f"[SUCCESS] บันทึกรูปภาพ Confusion Matrix เรียบร้อยที่: {save_path}")
    plt.show()

if __name__ == "__main__":
    evaluate()