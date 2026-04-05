import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def create_dataloaders(train_dir, test_dir, transform, batch_size):
    # โหลดข้อมูลจากโฟลเดอร์โดยตรง (ชื่อโฟลเดอร์คือชื่อคลาส)
    train_data = datasets.ImageFolder(root=train_dir, transform=transform)
    test_data = datasets.ImageFolder(root=test_dir, transform=transform)

    # ดึงรายชื่อคลาสสินค้าออกมา
    class_names = train_data.classes

    # สร้าง DataLoader
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True
    )
    
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True
    )

    return train_dataloader, test_dataloader, class_names