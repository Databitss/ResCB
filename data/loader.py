import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision import datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2

def loader(path_data, batch_size):
    # Location of Dataset
    data_dir = path_data
    raw_dataset = datasets.ImageFolder(root=data_dir)

    # Extract labels for stratification split format
    labels = [label for _, label in raw_dataset.samples]

    # Stratified split: 80% train, 20% test
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(np.zeros(len(labels)), labels))

    # Subset dataset sesuai dengan indeks yang diperoleh dari stratifikasi
    raw_train_dataset = Subset(raw_dataset, train_idx)
    raw_test_dataset = Subset(raw_dataset, test_idx)

    # Augmentasi untuk dataset training
    train_transform = A.Compose([
        A.Rotate(limit=20, p=0.5),  # Rotasi terbatas ±20°
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5), # Kontras dan kecerahan
        A.GaussianBlur(blur_limit=(3, 3), p=0.3),  # Blur ringan
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, p=0.5), # Hue dan saturasi
        A.HorizontalFlip(p=0.5), # Membalik secara horizontal
        A.VerticalFlip(p=0.5), # Membalik secara vertikal
        A.Resize(224, 224),  # Resize agar semua gambar memiliki ukuran yang sama
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Normalisasi
        ToTensorV2()  # Mengubah ke bentuk tensor
    ])

    # Untuk dataset validasi/test, hanya dilakukan normalisasi tanpa augmentasi
    test_transform = A.Compose([
        A.Resize(224, 224),  # Resize ke 224x224
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Normalisasi
        ToTensorV2(),  # Konversi ke tensor PyTorch
    ])

    # Dataset wrapper untuk augmentasi on-the-fly
    class AlbumentationsDataset(Dataset):
        def __init__(self, dataset, transform=None):
            self.dataset = dataset
            self.transform = transform

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            image, label = self.dataset[idx]
            image = np.array(image)  # Convert PIL image to numpy array
            
            if self.transform:
                augmented = self.transform(image=image) # Doing augmentation
                image = augmented['image'] 
            
            return image, label

    # Terapkan augmentasi hanya pada dataset training
    train_dataset = AlbumentationsDataset(raw_train_dataset, transform=train_transform)  # Train dataset dengan augmentasi
    test_dataset = AlbumentationsDataset(raw_test_dataset, transform=test_transform)  # Test dataset tanpa augmentasi

    # DataLoader untuk training & testing
    # batch_size = 32 # Jumlah batch size yang digunakan karna dataset kecil
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True) # Data train sudah siap 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True) # Data test sudah siap
    return train_loader, test_loader
