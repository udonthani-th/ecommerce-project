import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

# --- แก้ไขจุดที่ 1: Import จากโฟลเดอร์ src ---
from src import data_setup, model_builder, engine, utils

def main():
    print("[DEBUG] เริ่มรันไฟล์ train.py...")
    
    # ตั้งค่าอุปกรณ์
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEBUG] ใช้ Device: {DEVICE}")

    # 2. การจัดการข้อมูล (Data Transformation)
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. โหลด DataLoader
    print("[DEBUG] กำลังโหลด DataLoader...")
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir="data/train", 
        test_dir="data/test", 
        transform=data_transform, 
        batch_size=32
    )
    print(f"[DEBUG] โหลดเสร็จแล้ว! พบสินค้า {len(class_names)} ประเภท")

    # 4. สร้างโมเดล
    print("[DEBUG] กำลังสร้าง Model...")
    model = model_builder.create_model(num_classes=len(class_names)).to(DEVICE)

    # 5. ตั้งค่า Loss และ Optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 6. ตั้งค่า TensorBoard (บันทึกไว้ในโฟลเดอร์ runs)
    writer = SummaryWriter(log_dir="runs")

    # 7. เริ่มการฝึกฝน
    print(f"[DEBUG] เริ่มต้นการ Train...")
    # เก็บผลลัพธ์ไว้ในตัวแปร results เพื่อเอาไปพล็อตกราฟต่อได้
    results = engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=10,
        device=DEVICE,
        writer=writer
    )

    # --- แก้ไขจุดที่ 2: ใช้ฟังก์ชันจาก utils.py แทนการเขียน manual ---
    # 8. การบันทึกโมเดล
    utils.save_model(
        model=model,
        target_dir="models",
        model_name="ecommerce_model.pth"
    )
    
    # (ทางเลือกเพิ่มเติม) 9. พล็อตกราฟ Loss/Accuracy ทันทีหลังเทรนเสร็จ
    # utils.plot_loss_curves(results)
    
    print("[SUCCESS] การทำงานใน train.py เสร็จสิ้น!")

if __name__ == '__main__':
    main()