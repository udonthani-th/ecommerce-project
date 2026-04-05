# E-commerce Product Classification (Modular Implementation)
โปรเจกต์ Deep Learning สำหรับจำแนกประเภทสินค้าอัตโนมัติ โดยใช้เทคนิค 
  **Transfer Learning** บนสถาปัตยกรรม **ResNet50** ##  จุดเด่นของโปรเจกต์
* **Modular Structure:** แยกส่วนการทำงานของโค้ด (Data Setup, Model Building, Training Engine) เพื่อให้ง่ายต่อการบำรุงรักษา
* **Transfer Learning:** ใช้ Pre-trained weights จาก ImageNet เพื่อความแม่นยำสูงแม้มีข้อมูลจำกัด
* **Experiment Tracking:** รองรับการเก็บ Log ผ่าน TensorBoard เพื่อดูความก้าวหน้าของการเทรน
---
##  โครงสร้างโปรเจกต์ (Project Structure)
โปรเจกต์นี้ถูกออกแบบตามมาตรฐาน Modular Design ดังนี้:

my_ecommerce_project/
├── data/                # ชุดข้อมูลรูปภาพสินค้า
│   ├── train/           # ข้อมูลสำหรับการฝึกฝน (แบ่งโฟลเดอร์ตามคลาส)
│   └── test/            # ข้อมูลสำหรับการทดสอบ
├── models/              # โฟลเดอร์เก็บไฟล์โมเดล (.pth) ที่เทรนสำเร็จแล้ว
├── results/             # เก็บผลลัพธ์การประเมิน (เช่น Confusion Matrix)
├── runs/                # TensorBoard Logs สำหรับติดตามผล Accuracy/Loss
├── src/                 # ซอร์สโค้ดหลัก (Modules)
│   ├── __init__.py      # ไฟล์สำหรับกำหนดให้ src เป็น Python Package
│   ├── data_setup.py    # จัดการ Image DataLoader และการทำ Data Augmentation
│   ├── model_builder.py # สถาปัตยกรรมโมเดล ResNet50 และ Custom Classifier
│   ├── engine.py        # ฟังก์ชันหลักสำหรับ Training และ Testing Loop
│   └── utils.py         # ฟังก์ชันเสริมสำหรับบันทึกโมเดลและพล็อตกราฟ
├── train.py             # ไฟล์หลักสำหรับสั่งรันการเทรน (Main Entry Point)
├── evaluate.py          # ไฟล์สำหรับวัดผลประสิทธิภาพและสร้าง Confusion Matrix
├── requirements.txt     # รายการ Library และเวอร์ชันที่จำเป็น
└── .gitignore           # กำหนดไฟล์ที่ไม่ต้องการนำขึ้น GitHub (เช่น data, venv)
