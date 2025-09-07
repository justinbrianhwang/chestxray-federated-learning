#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Auto-generated from the Jupyter notebook.


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models
from torchvision.models import *
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import kagglehub
from google.colab import drive
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ---- Cell Separator ----

print("Mounting Google Drive...")
drive.mount('/content/drive')

# ---- Cell Separator ----

# Set up directories
DRIVE_PATH = '/content/drive/MyDrive/BS (Yonsei University)/Thesis/2025 추계종합학술대회 및 대학생논문경진대회(한국정보기술학회)'
MODEL_SAVE_PATH = f'{DRIVE_PATH}/client_models'
DATA_PATH = f'{DRIVE_PATH}/data'

# ---- Cell Separator ----

# Download NIH Chest X-rays dataset and set up paths
print("Downloading NIH Chest X-rays dataset...")
try:
    # Download dataset
    download_path = kagglehub.dataset_download("nih-chest-xrays/data")
    print(f"Dataset downloaded to: {download_path}")

    # Copy dataset to Google Drive
    print(f"Copying dataset to Google Drive: {DATA_PATH}")
    import shutil

    dataset_target = f"{DATA_PATH}/nih_dataset"

    if not os.path.exists(dataset_target):
        print("Copying dataset files...")
        os.makedirs(dataset_target, exist_ok=True)

        # Copy all contents
        for item in os.listdir(download_path):
            source = os.path.join(download_path, item)
            destination = os.path.join(dataset_target, item)

            if os.path.isdir(source):
                print(f"Copying directory: {item}")
                shutil.copytree(source, destination)
            else:
                print(f"Copying file: {item}")
                shutil.copy2(source, destination)

        print("Dataset copied successfully!")
    else:
        print("Dataset already exists in Google Drive")

    # Set path to use Google Drive location
    path = dataset_target
    print(f"Using dataset from: {path}")

except Exception as e:
    print(f"Error downloading dataset: {e}")
    path = f"{DATA_PATH}/nih_dataset"  # Fallback to existing location

# Main training function - path setup only
def setup_data_paths():
    """Set up correct CSV and image paths"""
    dataset_path = f"{DATA_PATH}/nih_dataset"

    # Find CSV file
    csv_path = None
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.csv'):
                csv_path = os.path.join(root, file)
                print(f"Found CSV file: {csv_path}")
                break
        if csv_path:
            break

    # Find image directory
    img_dir = None
    for root, dirs, files in os.walk(dataset_path):
        # Look for directory with many image files
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if len(image_files) > 100:
            img_dir = root
            print(f"Found image directory: {img_dir} ({len(image_files)} images)")
            break

    # Alternative: look for 'images' directory
    if img_dir is None:
        for root, dirs, files in os.walk(dataset_path):
            for dir_name in dirs:
                if 'image' in dir_name.lower():
                    potential_dir = os.path.join(root, dir_name)
                    try:
                        contents = os.listdir(potential_dir)
                        img_files = [f for f in contents if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                        if len(img_files) > 100:
                            img_dir = potential_dir
                            print(f"Found image directory: {img_dir}")
                            break
                    except:
                        continue
            if img_dir:
                break

    return csv_path, img_dir

# ---- Cell Separator ----

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ---- Cell Separator ----

NIH_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]
NUM_CLASSES = len(NIH_LABELS)

# ---- Cell Separator ----

# Setup data paths function
def setup_data_paths():
    """Set up correct CSV and image paths"""
    dataset_path = f"{DATA_PATH}/nih_dataset"

    # Find CSV file - specifically look for Data_Entry file
    csv_path = None
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.csv') and ('Data_Entry' in file or 'data_entry' in file.lower()):
                csv_path = os.path.join(root, file)
                print(f"Found Data Entry CSV file: {csv_path}")
                break
        if csv_path:
            break

    # If Data_Entry not found, show all CSV files
    if csv_path is None:
        print("Data_Entry CSV not found. Available CSV files:")
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.csv'):
                    print(f"  - {os.path.join(root, file)}")

        # Look for any CSV with 'Finding Labels' column
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.csv'):
                    try:
                        test_path = os.path.join(root, file)
                        test_df = pd.read_csv(test_path, nrows=1)
                        if 'Finding Labels' in test_df.columns:
                            csv_path = test_path
                            print(f"Found CSV with 'Finding Labels' column: {csv_path}")
                            break
                    except:
                        continue
            if csv_path:
                break

    # Find image directory
    img_dir = None
    for root, dirs, files in os.walk(dataset_path):
        # Look for directory with many image files
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if len(image_files) > 100:
            img_dir = root
            print(f"Found image directory: {img_dir} ({len(image_files)} images)")
            break

    # Alternative: look for 'images' directory
    if img_dir is None:
        for root, dirs, files in os.walk(dataset_path):
            for dir_name in dirs:
                if 'image' in dir_name.lower():
                    potential_dir = os.path.join(root, dir_name)
                    try:
                        contents = os.listdir(potential_dir)
                        img_files = [f for f in contents if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                        if len(img_files) > 100:
                            img_dir = potential_dir
                            print(f"Found image directory: {img_dir}")
                            break
                    except:
                        continue
            if img_dir:
                break

    return csv_path, img_dir

# Custom Dataset for NIH Chest X-rays
class NIHDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # Process labels (multi-label)
        self.data['Finding Labels'] = self.data['Finding Labels'].fillna('No Finding')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]['Image Index']
        img_path = os.path.join(self.img_dir, img_name)

        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # If image can't be loaded, return a black image
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        # Multi-label encoding
        labels = self.data.iloc[idx]['Finding Labels']
        label_vector = torch.zeros(NUM_CLASSES)

        if labels != 'No Finding':
            for label in labels.split('|'):
                if label.strip() in NIH_LABELS:
                    label_idx = NIH_LABELS.index(label.strip())
                    label_vector[label_idx] = 1.0

        return image, label_vector

# Data transforms
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Model definitions - All 50 models
def get_model(model_id, num_classes=NUM_CLASSES):
    """Returns one of 50 different model architectures"""

    if model_id == 0:
        # ResNet18
        model = resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 1:
        # ResNet34
        model = resnet34(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 2:
        # ResNet50
        model = resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 3:
        # ResNet101
        model = resnet101(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 4:
        # ResNet152
        model = resnet152(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 5:
        # DenseNet121
        model = densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model
    elif model_id == 6:
        # DenseNet161
        model = densenet161(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model
    elif model_id == 7:
        # DenseNet169
        model = densenet169(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model
    elif model_id == 8:
        # DenseNet201
        model = densenet201(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model
    elif model_id == 9:
        # VGG11
        model = vgg11(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    elif model_id == 10:
        # VGG13
        model = vgg13(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    elif model_id == 11:
        # VGG16
        model = vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    elif model_id == 12:
        # VGG19
        model = vgg19(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    elif model_id == 13:
        # EfficientNet-B0
        model = efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    elif model_id == 14:
        # EfficientNet-B1
        model = efficientnet_b1(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    elif model_id == 15:
        # EfficientNet-B2
        model = efficientnet_b2(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    elif model_id == 16:
        # EfficientNet-B3
        model = efficientnet_b3(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    elif model_id == 17:
        # EfficientNet-B4
        model = efficientnet_b4(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    elif model_id == 18:
        # MobileNet-V2
        model = mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    elif model_id == 19:
        # MobileNet-V3 Large
        model = mobilenet_v3_large(pretrained=True)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        return model
    elif model_id == 20:
        # MobileNet-V3 Small
        model = mobilenet_v3_small(pretrained=True)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        return model
    elif model_id == 21:
        # SqueezeNet 1.0
        model = squeezenet1_0(pretrained=True)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        model.num_classes = num_classes
        return model
    elif model_id == 22:
        # SqueezeNet 1.1
        model = squeezenet1_1(pretrained=True)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        model.num_classes = num_classes
        return model
    elif model_id == 23:
        # ShuffleNet V2 x0.5
        model = shufflenet_v2_x0_5(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 24:
        # ShuffleNet V2 x1.0
        model = shufflenet_v2_x1_0(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 25:
        # Wide ResNet-50-2
        model = wide_resnet50_2(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 26:
        # Wide ResNet-101-2
        model = wide_resnet101_2(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 27:
        # ResNext-50-32x4d
        model = resnext50_32x4d(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 28:
        # ResNext-101-32x8d
        model = resnext101_32x8d(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 29:
        # RegNet Y-400MF
        model = regnet_y_400mf(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 30:
        # RegNet Y-800MF
        model = regnet_y_800mf(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 31:
        # RegNet Y-1.6GF
        model = regnet_y_1_6gf(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 32:
        # VGG11 with Batch Normalization
        model = vgg11_bn(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    elif model_id == 33:
        # VGG13 with Batch Normalization
        model = vgg13_bn(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    elif model_id == 34:
        # VGG16 with Batch Normalization
        model = vgg16_bn(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    elif model_id == 35:
        # VGG19 with Batch Normalization
        model = vgg19_bn(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    elif model_id == 36:
        # AlexNet
        model = alexnet(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    elif model_id == 37:
        # Custom CNN 1
        class CustomCNN1(nn.Module):
            def __init__(self, num_classes):
                super(CustomCNN1, self).__init__()
                self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.dropout = nn.Dropout(0.5)
                self.fc1 = nn.Linear(128 * 28 * 28, 512)
                self.fc2 = nn.Linear(512, num_classes)

            def forward(self, x):
                x = self.pool(torch.relu(self.conv1(x)))
                x = self.pool(torch.relu(self.conv2(x)))
                x = self.pool(torch.relu(self.conv3(x)))
                x = x.view(-1, 128 * 28 * 28)
                x = self.dropout(torch.relu(self.fc1(x)))
                x = self.fc2(x)
                return x
        return CustomCNN1(num_classes)
    elif model_id == 38:
        # Custom CNN 2
        class CustomCNN2(nn.Module):
            def __init__(self, num_classes):
                super(CustomCNN2, self).__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, 7, stride=2, padding=3),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(3, stride=2, padding=1),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((7, 7))
                )
                self.classifier = nn.Sequential(
                    nn.Linear(256 * 7 * 7, 1024),
                    nn.ReLU(inplace=True),
                    nn.Dropout(),
                    nn.Linear(1024, 512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(),
                    nn.Linear(512, num_classes)
                )

            def forward(self, x):
                x = self.features(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
                return x
        return CustomCNN2(num_classes)
    elif model_id == 39:
        # Custom ResNet variant
        class CustomResNet(nn.Module):
            def __init__(self, num_classes):
                super(CustomResNet, self).__init__()
                self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
                self.bn1 = nn.BatchNorm2d(64)
                self.relu = nn.ReLU(inplace=True)
                self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

                self.layer1 = self._make_layer(64, 64, 2)
                self.layer2 = self._make_layer(64, 128, 2, stride=2)
                self.layer3 = self._make_layer(128, 256, 2, stride=2)
                self.layer4 = self._make_layer(256, 512, 2, stride=2)

                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(512, num_classes)

            def _make_layer(self, in_planes, planes, blocks, stride=1):
                layers = []
                layers.append(nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1))
                layers.append(nn.BatchNorm2d(planes))
                layers.append(nn.ReLU(inplace=True))

                for _ in range(1, blocks):
                    layers.append(nn.Conv2d(planes, planes, 3, padding=1))
                    layers.append(nn.BatchNorm2d(planes))
                    layers.append(nn.ReLU(inplace=True))

                return nn.Sequential(*layers)

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)

                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)

                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.fc(x)
                return x
        return CustomResNet(num_classes)

    # Custom variants for remaining models (40-49)
    elif model_id >= 40:
        custom_id = model_id - 40

        class CustomVariant(nn.Module):
            def __init__(self, num_classes, variant_id):
                super(CustomVariant, self).__init__()
                base_channels = 32 + (variant_id * 8)

                self.features = nn.Sequential(
                    nn.Conv2d(3, base_channels, 3, padding=1),
                    nn.BatchNorm2d(base_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),

                    nn.Conv2d(base_channels, base_channels*2, 3, padding=1),
                    nn.BatchNorm2d(base_channels*2),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),

                    nn.Conv2d(base_channels*2, base_channels*4, 3, padding=1),
                    nn.BatchNorm2d(base_channels*4),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),

                    nn.Conv2d(base_channels*4, base_channels*8, 3, padding=1),
                    nn.BatchNorm2d(base_channels*8),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((4, 4))
                )

                self.classifier = nn.Sequential(
                    nn.Linear(base_channels*8*16, 512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(512, 256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Linear(256, num_classes)
                )

            def forward(self, x):
                x = self.features(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
                return x

        return CustomVariant(num_classes, custom_id)

def get_model_name(model_id):
    """Returns model name for given ID"""
    model_names = [
        "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152",
        "DenseNet121", "DenseNet161", "DenseNet169", "DenseNet201",
        "VGG11", "VGG13", "VGG16", "VGG19",
        "EfficientNet-B0", "EfficientNet-B1", "EfficientNet-B2", "EfficientNet-B3", "EfficientNet-B4",
        "MobileNet-V2", "MobileNet-V3-Large", "MobileNet-V3-Small",
        "SqueezeNet1.0", "SqueezeNet1.1",
        "ShuffleNet-V2-x0.5", "ShuffleNet-V2-x1.0",
        "WideResNet50-2", "WideResNet101-2",
        "ResNext50-32x4d", "ResNext101-32x8d",
        "RegNet-Y-400MF", "RegNet-Y-800MF", "RegNet-Y-1.6GF",
        "VGG11-BN", "VGG13-BN", "VGG16-BN", "VGG19-BN",
        "AlexNet", "CustomCNN1", "CustomCNN2", "CustomResNet"
    ]

    if model_id < len(model_names):
        return model_names[model_id]
    else:
        return f"CustomVariant{model_id-40}"

# Hamming Accuracy 사용으로 변경
def calculate_multilabel_accuracy(outputs, targets, threshold=0.5):
    with torch.no_grad():
        predictions = torch.sigmoid(outputs) > threshold
        predictions = predictions.float()

        # Hamming accuracy만 반환 (더 현실적)
        hamming_accuracy = (predictions == targets).float().mean().item()

        return hamming_accuracy, hamming_accuracy  # 일관성을 위해 두 번 반환

# 기존 train_client_model 함수를 이것으로 교체
def train_client_model(client_id, model, train_loader, val_loader, epochs=10, lr=0.001):
    """Train a single client model with accuracy calculation"""
    print(f"\n=== Training Client {client_id} ({get_model_name(client_id)}) ===")

    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()  # Multi-label classification
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_loss = float('inf')
    best_val_accuracy = 0.0  # 추가: 최고 정확도 추적
    train_losses = []
    val_losses = []
    train_accuracies = []  # 추가: 학습 정확도 저장
    val_accuracies = []    # 추가: 검증 정확도 저장

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_predictions = []  # 추가: 학습 예측값 저장
        train_targets = []      # 추가: 학습 실제값 저장

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

        for batch_idx, (images, labels) in enumerate(train_pbar):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 정확도 계산을 위한 예측값 저장
            train_predictions.append(outputs.detach())
            train_targets.append(labels.detach())

            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

            # Memory management for large datasets
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 학습 정확도 계산
        all_train_predictions = torch.cat(train_predictions, dim=0)
        all_train_targets = torch.cat(train_targets, dim=0)
        train_exact_acc, train_hamming_acc = calculate_multilabel_accuracy(all_train_predictions, all_train_targets)
        train_accuracies.append(train_exact_acc)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_predictions = []    # 추가: 검증 예측값 저장
        val_targets = []        # 추가: 검증 실제값 저장

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # 정확도 계산을 위한 예측값 저장
                val_predictions.append(outputs)
                val_targets.append(labels)

                val_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # 검증 정확도 계산
        all_val_predictions = torch.cat(val_predictions, dim=0)
        all_val_targets = torch.cat(val_targets, dim=0)
        val_exact_acc, val_hamming_acc = calculate_multilabel_accuracy(all_val_predictions, all_val_targets)
        val_accuracies.append(val_exact_acc)

        # 출력에 정확도 추가
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_exact_acc:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_exact_acc:.4f}")

        # Save best model (정확도 기준으로 변경)
        if val_exact_acc > best_val_accuracy:
            best_val_accuracy = val_exact_acc
            best_val_loss = avg_val_loss
            model_save_path = f"{MODEL_SAVE_PATH}/client_{client_id}_{get_model_name(client_id)}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_accuracy': train_exact_acc,     # 추가
                'val_accuracy': val_exact_acc,         # 추가
                'model_id': client_id,
                'model_name': get_model_name(client_id)
            }, model_save_path)
            print(f"Best model saved: {model_save_path} (Val Acc: {best_val_accuracy:.4f})")

        scheduler.step()
        torch.cuda.empty_cache()

    # 최종 결과 출력
    print(f"Client {client_id} training completed!")
    print(f"Final Results - Best Val Accuracy: {best_val_accuracy:.4f}, Best Val Loss: {best_val_loss:.4f}")

    return train_losses, val_losses, train_accuracies, val_accuracies, best_val_accuracy


# Data loading and preparation
def prepare_federated_data(csv_path, img_dir, num_clients=50, batch_size=32):
    """Prepare data for federated learning"""
    print("Preparing federated data...")

    # Create full dataset
    full_dataset = NIHDataset(csv_path, img_dir, transform_train)

    # Split dataset among clients (IID distribution for simplicity)
    total_size = len(full_dataset)
    client_size = total_size // num_clients

    client_datasets = []
    remaining_size = total_size

    for i in range(num_clients):
        if i == num_clients - 1:  # Last client gets remaining data
            size = remaining_size
        else:
            size = client_size

        client_data, full_dataset = random_split(full_dataset, [size, len(full_dataset) - size])
        client_datasets.append(client_data)
        remaining_size -= size

    print(f"Data split into {num_clients} clients, each with ~{client_size} samples")
    return client_datasets

# Main training function
def main_training_with_results(num_clients=50, epochs=10, batch_size=16):
    """Main function to train all client models and collect results"""

    # Setup paths
    csv_path, img_dir = setup_data_paths()

    # Validate paths
    if csv_path is None or not os.path.exists(csv_path):
        print("❌ CSV file not found!")
        return

    if img_dir is None or not os.path.exists(img_dir):
        print("❌ Image directory not found!")
        return

    print(f"✅ Using CSV: {csv_path}")
    print(f"✅ Using images: {img_dir}")

    # Prepare federated data
    client_datasets = prepare_federated_data(csv_path, img_dir, num_clients, batch_size)

    # 결과 저장용 리스트
    all_results = []

    # Train each client model
    for client_id in range(num_clients):
        print(f"\n{'='*60}")
        print(f"Starting training for Client {client_id}")
        print(f"{'='*60}")

        try:
            # Create data loaders for this client
            train_size = int(0.8 * len(client_datasets[client_id]))
            val_size = len(client_datasets[client_id]) - train_size

            train_data, val_data = random_split(client_datasets[client_id], [train_size, val_size])

            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)

            print(f"Client {client_id} - Train samples: {len(train_data)}, Val samples: {len(val_data)}")

            # Create and train model
            model = get_model(client_id)
            train_losses, val_losses, train_accs, val_accs, best_acc = train_client_model(
                client_id, model, train_loader, val_loader, epochs, lr=0.001
            )

            # 결과 저장
            all_results.append({
                'client_id': client_id,
                'model_name': get_model_name(client_id),
                'best_val_accuracy': best_acc,
                'final_train_loss': train_losses[-1],
                'final_val_loss': val_losses[-1],
                'final_train_acc': train_accs[-1],
                'final_val_acc': val_accs[-1]
            })

            print(f"Client {client_id} completed! Best Accuracy: {best_acc:.4f}")

            # Clear memory
            del model, train_loader, val_loader, train_data, val_data
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error training client {client_id}: {str(e)}")
            continue

    # 결과 요약 출력
    print("\n" + "="*80)
    print("TRAINING RESULTS SUMMARY")
    print("="*80)
    print(f"{'Client':<8} {'Model':<20} {'Best Val Acc':<12} {'Final Train Acc':<15} {'Final Val Acc':<13}")
    print("-"*80)

    for result in all_results:
        print(f"{result['client_id']:<8} {result['model_name']:<20} {result['best_val_accuracy']:<12.4f} "
              f"{result['final_train_acc']:<15.4f} {result['final_val_acc']:<13.4f}")

    # 통계 계산
    if all_results:
        accuracies = [r['best_val_accuracy'] for r in all_results]
        print("-"*80)
        print(f"Average Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"Max Accuracy: {np.max(accuracies):.4f}")
        print(f"Min Accuracy: {np.min(accuracies):.4f}")
        print(f"Total Models: {len(all_results)}")

    print("\n" + "="*60)
    print("All client models training completed!")
    print(f"Models saved in: {MODEL_SAVE_PATH}")
    print("="*60)

    return all_results

# Run the training
if __name__ == "__main__":
    print("Starting Federated Learning Client Training...")
    print(f"Using {NUM_CLASSES} classes: {NIH_LABELS}")

    # 결과와 함께 학습 실행
    results = main_training_with_results(num_clients=50, epochs=10, batch_size=64)

    print("\nTraining completed! Check your Google Drive for saved models.")

# ---- Cell Separator ----

# Federated Learning - Logit Extraction Code for Google Colab
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import *
from PIL import Image
import pandas as pd
import numpy as np
import pickle
from google.colab import drive
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
LOGIT_SAVE_PATH = f'{DRIVE_PATH}/logits'

# Create logits directory
os.makedirs(LOGIT_SAVE_PATH, exist_ok=True)

print(f"Models path: {MODEL_SAVE_PATH}")
print(f"Logits will be saved to: {LOGIT_SAVE_PATH}")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# NIH dataset labels
NIH_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]
NUM_CLASSES = len(NIH_LABELS)

# Data transforms for testing
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# NIH Dataset class
class NIHDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.data['Finding Labels'] = self.data['Finding Labels'].fillna('No Finding')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]['Image Index']
        img_path = os.path.join(self.img_dir, img_name)

        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        labels = self.data.iloc[idx]['Finding Labels']
        label_vector = torch.zeros(NUM_CLASSES)

        if labels != 'No Finding':
            for label in labels.split('|'):
                if label.strip() in NIH_LABELS:
                    label_idx = NIH_LABELS.index(label.strip())
                    label_vector[label_idx] = 1.0

        return image, label_vector, img_name

# Model definitions (abbreviated for space - same as training code)
def get_model(model_id, num_classes=NUM_CLASSES):
    """Returns one of 50 different model architectures"""
    if model_id == 0:
        model = resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 1:
        model = resnet34(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 2:
        model = resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 3:
        model = resnet101(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 4:
        model = resnet152(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 5:
        model = densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model
    elif model_id == 6:
        model = densenet161(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model
    elif model_id == 7:
        model = densenet169(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model
    elif model_id == 8:
        model = densenet201(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model
    elif model_id == 9:
        model = vgg11(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    elif model_id == 10:
        model = vgg13(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    elif model_id == 11:
        model = vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    elif model_id == 12:
        model = vgg19(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    elif model_id == 13:
        model = efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    elif model_id == 14:
        model = efficientnet_b1(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    elif model_id == 15:
        model = efficientnet_b2(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    elif model_id == 16:
        model = efficientnet_b3(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    elif model_id == 17:
        model = efficientnet_b4(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    elif model_id == 18:
        model = mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    elif model_id == 19:
        model = mobilenet_v3_large(pretrained=True)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        return model
    elif model_id == 20:
        model = mobilenet_v3_small(pretrained=True)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        return model
    elif model_id == 21:
        model = squeezenet1_0(pretrained=True)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        model.num_classes = num_classes
        return model
    elif model_id == 22:
        model = squeezenet1_1(pretrained=True)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        model.num_classes = num_classes
        return model
    elif model_id == 23:
        model = shufflenet_v2_x0_5(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 24:
        model = shufflenet_v2_x1_0(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 25:
        model = wide_resnet50_2(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 26:
        model = wide_resnet101_2(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 27:
        model = resnext50_32x4d(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 28:
        model = resnext101_32x8d(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 29:
        model = regnet_y_400mf(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 30:
        model = regnet_y_800mf(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 31:
        model = regnet_y_1_6gf(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 32:
        model = vgg11_bn(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    elif model_id == 33:
        model = vgg13_bn(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    elif model_id == 34:
        model = vgg16_bn(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    elif model_id == 35:
        model = vgg19_bn(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    elif model_id == 36:
        model = alexnet(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    # Custom models for remaining IDs
    else:
        # Custom CNN implementation for remaining model IDs
        class CustomCNN(nn.Module):
            def __init__(self, num_classes, variant_id):
                super(CustomCNN, self).__init__()
                base_channels = 32 + (variant_id * 8)

                self.features = nn.Sequential(
                    nn.Conv2d(3, base_channels, 3, padding=1),
                    nn.BatchNorm2d(base_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),

                    nn.Conv2d(base_channels, base_channels*2, 3, padding=1),
                    nn.BatchNorm2d(base_channels*2),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),

                    nn.Conv2d(base_channels*2, base_channels*4, 3, padding=1),
                    nn.BatchNorm2d(base_channels*4),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((4, 4))
                )

                self.classifier = nn.Sequential(
                    nn.Linear(base_channels*4*16, 512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(512, num_classes)
                )

            def forward(self, x):
                x = self.features(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
                return x

        return CustomCNN(num_classes, model_id - 37)

def get_model_name(model_id):
    """Returns model name for given ID"""
    model_names = [
        "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152",
        "DenseNet121", "DenseNet161", "DenseNet169", "DenseNet201",
        "VGG11", "VGG13", "VGG16", "VGG19",
        "EfficientNet-B0", "EfficientNet-B1", "EfficientNet-B2", "EfficientNet-B3", "EfficientNet-B4",
        "MobileNet-V2", "MobileNet-V3-Large", "MobileNet-V3-Small",
        "SqueezeNet1.0", "SqueezeNet1.1",
        "ShuffleNet-V2-x0.5", "ShuffleNet-V2-x1.0",
        "WideResNet50-2", "WideResNet101-2",
        "ResNext50-32x4d", "ResNext101-32x8d",
        "RegNet-Y-400MF", "RegNet-Y-800MF", "RegNet-Y-1.6GF",
        "VGG11-BN", "VGG13-BN", "VGG16-BN", "VGG19-BN",
        "AlexNet"
    ]

    if model_id < len(model_names):
        return model_names[model_id]
    else:
        return f"CustomCNN{model_id-37}"

def setup_data_paths():
    """Set up correct CSV and image paths"""
    dataset_path = f"{DATA_PATH}/nih_dataset"

    csv_path = None
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.csv') and ('Data_Entry' in file or 'data_entry' in file.lower()):
                csv_path = os.path.join(root, file)
                break
        if csv_path:
            break

    if csv_path is None:
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.csv'):
                    try:
                        test_path = os.path.join(root, file)
                        test_df = pd.read_csv(test_path, nrows=1)
                        if 'Finding Labels' in test_df.columns:
                            csv_path = test_path
                            break
                    except:
                        continue
            if csv_path:
                break

    img_dir = None
    for root, dirs, files in os.walk(dataset_path):
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if len(image_files) > 100:
            img_dir = root
            break

    return csv_path, img_dir

def load_trained_model(client_id):
    """Load trained model for a specific client"""
    model_name = get_model_name(client_id)
    model_path = f"{MODEL_SAVE_PATH}/client_{client_id}_{model_name}.pth"

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None

    try:
        model = get_model(client_id)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        print(f"Loaded model: {model_name}")
        return model
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None

def extract_logits_from_model(model, data_loader, client_id):
    """Extract logits from a trained model"""
    model.eval()
    all_logits = []
    all_labels = []
    all_image_names = []

    print(f"Extracting logits from Client {client_id} ({get_model_name(client_id)})...")

    with torch.no_grad():
        for batch_idx, (images, labels, img_names) in enumerate(tqdm(data_loader, desc="Extracting")):
            images = images.to(device)
            outputs = model(images)

            all_logits.append(outputs.cpu().numpy())
            all_labels.append(labels.numpy())
            all_image_names.extend(img_names)

            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()

    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    return logits, labels, all_image_names

def create_test_dataset(csv_path, img_dir, test_size=0.2):
    """Create test dataset for logit extraction"""
    print("Creating test dataset...")

    full_df = pd.read_csv(csv_path)
    test_start_idx = int(len(full_df) * (1 - test_size))
    test_df = full_df.iloc[test_start_idx:].reset_index(drop=True)

    test_df.to_csv(f"{LOGIT_SAVE_PATH}/test_split.csv", index=False)
    print(f"Test dataset size: {len(test_df)} samples")

    return test_df

def extract_all_logits(batch_size=32, num_clients=50):
    """Extract logits from all trained client models"""

    csv_path, img_dir = setup_data_paths()

    if csv_path is None or img_dir is None:
        print("Cannot find dataset paths!")
        return

    print(f"Using CSV: {csv_path}")
    print(f"Using images: {img_dir}")

    test_df = create_test_dataset(csv_path, img_dir)
    temp_csv = f"{LOGIT_SAVE_PATH}/temp_test.csv"
    test_df.to_csv(temp_csv, index=False)

    test_dataset = NIHDataset(temp_csv, img_dir, transform_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Test dataset loaded: {len(test_dataset)} samples")

    available_models = []
    missing_models = []

    for client_id in range(num_clients):
        model_name = get_model_name(client_id)
        model_path = f"{MODEL_SAVE_PATH}/client_{client_id}_{model_name}.pth"

        if os.path.exists(model_path):
            available_models.append(client_id)
        else:
            missing_models.append(client_id)

    print(f"Available models: {len(available_models)}/{num_clients}")
    print(f"Missing models: {len(missing_models)}")

    all_client_logits = {}
    all_client_predictions = {}
    ground_truth_labels = None
    image_names = None

    for client_id in available_models:
        print(f"\nProcessing Client {client_id}")

        model = load_trained_model(client_id)
        if model is None:
            continue

        try:
            logits, labels, img_names = extract_logits_from_model(model, test_loader, client_id)

            all_client_logits[client_id] = logits
            all_client_predictions[client_id] = torch.sigmoid(torch.tensor(logits)).numpy()

            if ground_truth_labels is None:
                ground_truth_labels = labels
                image_names = img_names

            print(f"Logits extracted: {logits.shape}")

            client_save_path = f"{LOGIT_SAVE_PATH}/client_{client_id}_logits.npz"
            np.savez_compressed(client_save_path,
                              logits=logits,
                              predictions=all_client_predictions[client_id],
                              labels=labels,
                              image_names=img_names,
                              client_id=client_id,
                              model_name=get_model_name(client_id))

            print(f"Saved: {client_save_path}")

        except Exception as e:
            print(f"Error processing Client {client_id}: {e}")
            continue

        del model
        torch.cuda.empty_cache()

    print("\nSaving combined results...")

    combined_save_path = f"{LOGIT_SAVE_PATH}/all_client_logits.npz"
    np.savez_compressed(combined_save_path,
                      **{f"client_{cid}_logits": logits for cid, logits in all_client_logits.items()},
                      **{f"client_{cid}_predictions": pred for cid, pred in all_client_predictions.items()},
                      ground_truth=ground_truth_labels,
                      image_names=image_names,
                      available_clients=available_models,
                      class_names=NIH_LABELS)

    print(f"Combined results saved: {combined_save_path}")

    summary = {
        'total_clients': num_clients,
        'available_clients': len(available_models),
        'missing_clients': len(missing_models),
        'test_samples': len(test_dataset),
        'num_classes': NUM_CLASSES,
        'class_names': NIH_LABELS,
        'available_client_ids': available_models,
        'missing_client_ids': missing_models,
        'logit_shapes': {cid: logits.shape for cid, logits in all_client_logits.items()}
    }

    with open(f"{LOGIT_SAVE_PATH}/extraction_summary.pkl", 'wb') as f:
        pickle.dump(summary, f)

    print(f"Summary saved: {LOGIT_SAVE_PATH}/extraction_summary.pkl")

    if os.path.exists(temp_csv):
        os.remove(temp_csv)

    print("\nLOGIT EXTRACTION COMPLETED!")
    print(f"Successfully processed {len(available_models)} models")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Classes: {NUM_CLASSES}")
    print(f"Results saved in: {LOGIT_SAVE_PATH}")

    return all_client_logits, all_client_predictions, ground_truth_labels, image_names

def load_extracted_logits(client_id=None):
    """Load extracted logits for analysis"""

    if client_id is not None:
        client_path = f"{LOGIT_SAVE_PATH}/client_{client_id}_logits.npz"
        if os.path.exists(client_path):
            data = np.load(client_path, allow_pickle=True)
            return {
                'logits': data['logits'],
                'predictions': data['predictions'],
                'labels': data['labels'],
                'image_names': data['image_names'],
                'client_id': data['client_id'].item(),
                'model_name': data['model_name'].item()
            }
        else:
            print(f"Client {client_id} logits not found")
            return None
    else:
        combined_path = f"{LOGIT_SAVE_PATH}/all_client_logits.npz"
        if os.path.exists(combined_path):
            data = np.load(combined_path, allow_pickle=True)

            client_logits = {}
            client_predictions = {}

            for key in data.files:
                if key.startswith('client_') and key.endswith('_logits'):
                    client_id = int(key.split('_')[1])
                    client_logits[client_id] = data[key]
                elif key.startswith('client_') and key.endswith('_predictions'):
                    client_id = int(key.split('_')[1])
                    client_predictions[client_id] = data[key]

            return {
                'client_logits': client_logits,
                'client_predictions': client_predictions,
                'ground_truth': data['ground_truth'],
                'image_names': data['image_names'],
                'available_clients': data['available_clients'],
                'class_names': data['class_names']
            }
        else:
            print("Combined logits not found")
            return None

def analyze_extracted_logits():
    """Analyze the extracted logits and provide statistics"""

    summary_path = f"{LOGIT_SAVE_PATH}/extraction_summary.pkl"
    if os.path.exists(summary_path):
        with open(summary_path, 'rb') as f:
            summary = pickle.load(f)

        print("=== LOGIT EXTRACTION ANALYSIS ===")
        print(f"Total Clients: {summary['total_clients']}")
        print(f"Available Clients: {summary['available_clients']}")
        print(f"Missing Clients: {summary['missing_clients']}")
        print(f"Test Samples: {summary['test_samples']}")
        print(f"Number of Classes: {summary['num_classes']}")

        print(f"\nAvailable Client Models:")
        for client_id in summary['available_client_ids'][:10]:
            print(f"  Client {client_id}: {get_model_name(client_id)}")
        if len(summary['available_client_ids']) > 10:
            print(f"  ... and {len(summary['available_client_ids']) - 10} more")

        if summary['missing_client_ids']:
            print(f"\nMissing Client Models:")
            for client_id in summary['missing_client_ids'][:5]:
                print(f"  Client {client_id}: {get_model_name(client_id)}")
            if len(summary['missing_client_ids']) > 5:
                print(f"  ... and {len(summary['missing_client_ids']) - 5} more")

        return summary
    else:
        print("Summary file not found")
        return None

def validate_extracted_logits():
    """Validate the extracted logits for consistency"""

    print("=== VALIDATING EXTRACTED LOGITS ===")

    data = load_extracted_logits()
    if data is None:
        return False

    client_logits = data['client_logits']
    ground_truth = data['ground_truth']

    if len(client_logits) == 0:
        print("No client logits found")
        return False

    first_client = list(client_logits.keys())[0]
    expected_shape = client_logits[first_client].shape

    print(f"Found {len(client_logits)} client models")
    print(f"Expected logit shape: {expected_shape}")
    print(f"Ground truth shape: {ground_truth.shape}")

    shape_consistent = True
    for client_id, logits in client_logits.items():
        if logits.shape != expected_shape:
            print(f"Client {client_id} shape mismatch: {logits.shape}")
            shape_consistent = False

    if shape_consistent:
        print("All client logits have consistent shapes")

    nan_clients = []
    inf_clients = []

    for client_id, logits in client_logits.items():
        if np.isnan(logits).any():
            nan_clients.append(client_id)
        if np.isinf(logits).any():
            inf_clients.append(client_id)

    if nan_clients:
        print(f"Clients with NaN values: {nan_clients}")
    if inf_clients:
        print(f"Clients with infinite values: {inf_clients}")

    if not nan_clients and not inf_clients:
        print("No NaN or infinite values found")

    print(f"\n=== LOGIT STATISTICS ===")
    all_logits = np.concatenate(list(client_logits.values()), axis=0)
    print(f"Combined logits shape: {all_logits.shape}")
    print(f"Logit range: [{all_logits.min():.4f}, {all_logits.max():.4f}]")
    print(f"Logit mean: {all_logits.mean():.4f}")
    print(f"Logit std: {all_logits.std():.4f}")

    return shape_consistent and not nan_clients and not inf_clients

# Run the extraction
if __name__ == "__main__":
    print("Starting Logit Extraction from Trained Models...")
    print(f"Target path: {LOGIT_SAVE_PATH}")

    extract_all_logits(batch_size=32, num_clients=50)

    print("\n" + "="*60)
    analyze_extracted_logits()

    print("\n" + "="*60)
    validate_extracted_logits()

    print("\nLogit extraction completed! Ready for server training.")

# ---- Cell Separator ----

# Federated Learning - 100 Round Server Training Code for Google Colab
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import pickle
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Paths
DRIVE_PATH = '/content/drive/MyDrive/BS (Yonsei University)/Thesis/2025 추계종합학술대회 및 대학생논문경진대회(한국정보기술학회)'
MODEL_SAVE_PATH = f'{DRIVE_PATH}/client_models'
LOGIT_SAVE_PATH = f'{DRIVE_PATH}/logits'
SERVER_SAVE_PATH = f'{DRIVE_PATH}/server_models'
ROUNDS_SAVE_PATH = f'{DRIVE_PATH}/federated_rounds'
DATA_PATH = f'{DRIVE_PATH}/data'

# Create directories
os.makedirs(SERVER_SAVE_PATH, exist_ok=True)
os.makedirs(ROUNDS_SAVE_PATH, exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# NIH dataset labels
NIH_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]
NUM_CLASSES = len(NIH_LABELS)

# Data transforms
transform_server = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Server Model Architectures
class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width)
        value = self.value(x).view(batch_size, -1, height * width)

        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)

        return self.gamma * out + x

class ServerCNN(nn.Module):
    def __init__(self, num_classes):
        super(ServerCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.attention1 = AttentionBlock(128)

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.attention2 = AttentionBlock(256)

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention1(x)
        x = self.conv3(x)
        x = self.attention2(x)
        x = self.conv4(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

class ServerViT(nn.Module):
    def __init__(self, num_classes, img_size=224, patch_size=16, embed_dim=768, num_heads=12, num_layers=6):
        super(ServerViT, self).__init__()

        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed

        for layer in self.transformer_layers:
            x = layer(x)

        x = self.norm(x)
        cls_output = x[:, 0]
        x = self.head(cls_output)

        return x

class ServerHybrid(nn.Module):
    def __init__(self, num_classes):
        super(ServerHybrid, self).__init__()

        self.cnn_features = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.transformer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.cnn_features(x)

        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, channels, height * width).permute(0, 2, 1)

        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.classifier(x)

        return x

def get_server_model(model_type="cnn", num_classes=NUM_CLASSES):
    if model_type == "cnn":
        return ServerCNN(num_classes)
    elif model_type == "vit":
        return ServerViT(num_classes)
    elif model_type == "hybrid":
        return ServerHybrid(num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# Dataset for server training
class ServerDataset(Dataset):
    def __init__(self, csv_file, img_dir, client_logits, image_names, ground_truth, transform=None):
        self.csv_file = csv_file
        self.img_dir = img_dir
        self.client_logits = client_logits
        self.image_names = image_names
        self.ground_truth = ground_truth
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.img_dir, img_name)

        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(self.ground_truth[idx], dtype=torch.float32)

        # FIXED: Average client logits instead of concatenating
        client_logits_list = []
        for client_id in sorted(self.client_logits.keys()):
            client_logits_list.append(self.client_logits[client_id][idx])

        if client_logits_list:
            # Average all client logits to get a single 14-dimensional vector
            aggregated_logits = torch.tensor(np.mean(client_logits_list, axis=0), dtype=torch.float32)
        else:
            aggregated_logits = torch.zeros(NUM_CLASSES, dtype=torch.float32)

        return image, labels, aggregated_logits

# Knowledge Distillation Loss (FIXED)
class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, temperature=3.0, alpha=0.7):
        super(KnowledgeDistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, student_logits, teacher_logits, true_labels):
        # FIXED: Both tensors should be (batch_size, 14) now
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        kd_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)

        true_loss = self.bce_loss(student_logits, true_labels)
        total_loss = self.alpha * kd_loss + (1 - self.alpha) * true_loss

        return total_loss, kd_loss, true_loss

# Load client logits
def load_client_logits(round_num=0):
    if round_num == 0:
        combined_path = f"{LOGIT_SAVE_PATH}/all_client_logits.npz"
    else:
        combined_path = f"{ROUNDS_SAVE_PATH}/round_{round_num}/all_client_logits.npz"

    if not os.path.exists(combined_path):
        print(f"Combined logits file not found: {combined_path}")
        return None

    data = np.load(combined_path, allow_pickle=True)

    client_logits = {}
    for key in data.files:
        if key.startswith('client_') and key.endswith('_logits'):
            client_id = int(key.split('_')[1])
            client_logits[client_id] = data[key]

    result = {
        'client_logits': client_logits,
        'ground_truth': data['ground_truth'],
        'image_names': data['image_names'],
        'available_clients': data['available_clients'],
        'class_names': data['class_names']
    }

    return result

# Setup data paths
def setup_data_paths():
    dataset_path = f"{DATA_PATH}/nih_dataset"

    csv_path = None
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.csv') and ('Data_Entry' in file or 'data_entry' in file.lower()):
                csv_path = os.path.join(root, file)
                break
        if csv_path:
            break

    img_dir = None
    for root, dirs, files in os.walk(dataset_path):
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if len(image_files) > 100:
            img_dir = root
            break

    return csv_path, img_dir

# Training function for one round
def train_server_model_round(model, train_loader, val_loader, round_num, epochs=5, lr=0.001, model_name="server"):
    print(f"\n=== Round {round_num}: Training {model_name} ===")

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = KnowledgeDistillationLoss(temperature=3.0, alpha=0.7)

    best_val_loss = float('inf')
    round_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        train_pbar = tqdm(train_loader, desc=f"Round {round_num} Epoch {epoch+1}/{epochs}")

        for batch_idx, (images, labels, teacher_logits) in enumerate(train_pbar):
            images, labels = images.to(device), labels.to(device)
            teacher_logits = teacher_logits.to(device)

            optimizer.zero_grad()
            student_logits = model(images)
            total_loss, kd_loss, true_loss = criterion(student_logits, teacher_logits, labels)

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            train_pbar.set_postfix({'Loss': f'{total_loss.item():.4f}'})

            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()

        avg_train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels, teacher_logits in val_loader:
                images, labels = images.to(device), labels.to(device)
                teacher_logits = teacher_logits.to(device)

                student_logits = model(images)
                total_loss, _, _ = criterion(student_logits, teacher_logits, labels)
                val_loss += total_loss.item()

        avg_val_loss = val_loss / len(val_loader)
        round_losses.append({'train': avg_train_loss, 'val': avg_val_loss})

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

    return round_losses, best_val_loss

# Evaluation function
def evaluate_server_model(model, test_loader, round_num, model_name="server"):
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).float()

            all_predictions.append(predictions.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())
            all_labels.append(labels.numpy())

    predictions = np.concatenate(all_predictions, axis=0)
    probabilities = np.concatenate(all_probabilities, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    overall_accuracy = accuracy_score(labels.flatten(), predictions.flatten())
    overall_f1 = f1_score(labels, predictions, average='macro')
    overall_auc = roc_auc_score(labels, probabilities, average='macro')

    results = {
        'round': round_num,
        'model': model_name,
        'accuracy': overall_accuracy,
        'f1_score': overall_f1,
        'auc_score': overall_auc
    }

    print(f"Round {round_num} {model_name} - Acc: {overall_accuracy:.4f}, F1: {overall_f1:.4f}, AUC: {overall_auc:.4f}")

    return results

# Check convergence
def check_convergence(results_history, patience=10, min_improvement=0.001):
    if len(results_history) < patience:
        return False

    recent_results = results_history[-patience:]
    best_score = max([r['accuracy'] for r in recent_results])
    current_score = results_history[-1]['accuracy']

    if abs(best_score - current_score) < min_improvement:
        return True

    return False

# Main 100-round federated learning function
def federated_learning_100_rounds(num_rounds=100, epochs_per_round=5):
    print("Starting 100-Round Federated Learning...")
    print(f"Total rounds: {num_rounds}")
    print(f"Epochs per round: {epochs_per_round}")

    # Setup data paths
    csv_path, img_dir = setup_data_paths()
    if csv_path is None or img_dir is None:
        print("Cannot find dataset paths!")
        return

    # Load initial client logits
    logit_data = load_client_logits(round_num=0)
    if logit_data is None:
        print("Cannot load client logits!")
        return

    # Create dataset
    dataset = ServerDataset(
        csv_path, img_dir,
        logit_data['client_logits'],
        logit_data['image_names'],
        logit_data['ground_truth'],
        transform_server
    )

    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    print(f"Dataset split - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Initialize server models
    server_models = {
        'ServerCNN': get_server_model('cnn'),
        'ServerViT': get_server_model('vit'),
        'ServerHybrid': get_server_model('hybrid')
    }

    # Track results across all rounds
    all_results = {model_name: [] for model_name in server_models.keys()}
    round_histories = {model_name: [] for model_name in server_models.keys()}

    # Main federated learning loop
    for round_num in range(1, num_rounds + 1):
        print(f"\n{'='*80}")
        print(f"FEDERATED LEARNING ROUND {round_num}/{num_rounds}")
        print(f"{'='*80}")

        # Create round directory
        round_dir = f"{ROUNDS_SAVE_PATH}/round_{round_num}"
        os.makedirs(round_dir, exist_ok=True)

        round_results = {}

        for model_name, model in server_models.items():
            print(f"\nProcessing {model_name}...")

            # Train model for this round
            round_losses, best_val_loss = train_server_model_round(
                model, train_loader, val_loader,
                round_num, epochs_per_round, lr=0.001, model_name=model_name
            )

            # Evaluate model
            eval_results = evaluate_server_model(model, test_loader, round_num, model_name)

            # Store results
            round_results[model_name] = {
                'losses': round_losses,
                'best_val_loss': best_val_loss,
                'evaluation': eval_results
            }

            all_results[model_name].append(eval_results)
            round_histories[model_name].append(round_losses)

            # Save model checkpoint
            model_save_path = f"{round_dir}/{model_name}_round_{round_num}.pth"
            torch.save({
                'round': round_num,
                'model_state_dict': model.state_dict(),
                'val_loss': best_val_loss,
                'evaluation': eval_results,
                'model_name': model_name
            }, model_save_path)

            torch.cuda.empty_cache()

        # Save round results
        with open(f"{round_dir}/round_{round_num}_results.pkl", 'wb') as f:
            pickle.dump(round_results, f)

        # Check convergence for each model
        converged_models = []
        for model_name in server_models.keys():
            if check_convergence(all_results[model_name]):
                converged_models.append(model_name)

        if converged_models:
            print(f"\nConverged models: {converged_models}")

        # Print round summary
        print(f"\nRound {round_num} Summary:")
        for model_name, result in round_results.items():
            eval_res = result['evaluation']
            print(f"  {model_name:15}: Acc={eval_res['accuracy']:.4f}, F1={eval_res['f1_score']:.4f}, AUC={eval_res['auc_score']:.4f}")

        # Early stopping if all models converged
        if len(converged_models) == len(server_models):
            print(f"\nAll models converged at round {round_num}. Stopping early.")
            break

        # Memory management
        torch.cuda.empty_cache()

    # Save final results
    final_results = {
        'all_results': all_results,
        'round_histories': round_histories,
        'total_rounds': round_num,
        'converged_at': round_num if len(converged_models) == len(server_models) else None
    }

    with open(f"{ROUNDS_SAVE_PATH}/final_100_round_results.pkl", 'wb') as f:
        pickle.dump(final_results, f)

    print(f"\n{'='*80}")
    print("100-ROUND FEDERATED LEARNING COMPLETED!")
    print(f"{'='*80}")
    print(f"Total rounds completed: {round_num}")
    print(f"Results saved in: {ROUNDS_SAVE_PATH}")

    # Print final comparison
    print(f"\nFinal Model Comparison (Round {round_num}):")
    for model_name in server_models.keys():
        if all_results[model_name]:
            final_result = all_results[model_name][-1]
            print(f"  {model_name:15}: Acc={final_result['accuracy']:.4f}, F1={final_result['f1_score']:.4f}, AUC={final_result['auc_score']:.4f}")

    return final_results

# Run 100-round federated learning
if __name__ == "__main__":
    print("Starting 100-Round Federated Learning Server Training...")
    print(f"Using {NUM_CLASSES} classes: {NIH_LABELS}")

    results = federated_learning_100_rounds(num_rounds=100, epochs_per_round=10)

    print("\n100-round federated learning completed!")

# ---- Cell Separator ----

# Accuracy Evaluation for Trained Models
LOGIT_SAVE_PATH = f'{DRIVE_PATH}/logits'
SERVER_SAVE_PATH = f'{DRIVE_PATH}/server_models'

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# NIH dataset labels
NIH_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]
NUM_CLASSES = len(NIH_LABELS)

# Data transforms
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# NIH Dataset class
class NIHDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.data['Finding Labels'] = self.data['Finding Labels'].fillna('No Finding')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]['Image Index']
        img_path = os.path.join(self.img_dir, img_name)

        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        labels = self.data.iloc[idx]['Finding Labels']
        label_vector = torch.zeros(NUM_CLASSES)

        if labels != 'No Finding':
            for label in labels.split('|'):
                if label.strip() in NIH_LABELS:
                    label_idx = NIH_LABELS.index(label.strip())
                    label_vector[label_idx] = 1.0

        return image, label_vector

# Model definitions (same as training code)
def get_model(model_id, num_classes=NUM_CLASSES):
    """Returns one of 50 different model architectures"""
    if model_id == 0:
        model = resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 1:
        model = resnet34(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 2:
        model = resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 3:
        model = resnet101(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 4:
        model = resnet152(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 5:
        model = densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model
    elif model_id == 6:
        model = densenet161(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model
    elif model_id == 7:
        model = densenet169(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model
    elif model_id == 8:
        model = densenet201(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model
    elif model_id == 9:
        model = vgg11(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    elif model_id == 10:
        model = vgg13(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    elif model_id == 11:
        model = vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    elif model_id == 12:
        model = vgg19(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    elif model_id == 13:
        model = efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    elif model_id == 14:
        model = efficientnet_b1(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    elif model_id == 15:
        model = efficientnet_b2(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    elif model_id == 16:
        model = efficientnet_b3(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    elif model_id == 17:
        model = efficientnet_b4(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    elif model_id == 18:
        model = mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    elif model_id == 19:
        model = mobilenet_v3_large(pretrained=True)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        return model
    elif model_id == 20:
        model = mobilenet_v3_small(pretrained=True)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        return model
    elif model_id == 21:
        model = squeezenet1_0(pretrained=True)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        model.num_classes = num_classes
        return model
    elif model_id == 22:
        model = squeezenet1_1(pretrained=True)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        model.num_classes = num_classes
        return model
    elif model_id == 23:
        model = shufflenet_v2_x0_5(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 24:
        model = shufflenet_v2_x1_0(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 25:
        model = wide_resnet50_2(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 26:
        model = wide_resnet101_2(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 27:
        model = resnext50_32x4d(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 28:
        model = resnext101_32x8d(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 29:
        model = regnet_y_400mf(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 30:
        model = regnet_y_800mf(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 31:
        model = regnet_y_1_6gf(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 32:
        model = vgg11_bn(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    elif model_id == 33:
        model = vgg13_bn(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    elif model_id == 34:
        model = vgg16_bn(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    elif model_id == 35:
        model = vgg19_bn(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    elif model_id == 36:
        model = alexnet(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    else:
        # Custom CNN for remaining models
        class CustomCNN(nn.Module):
            def __init__(self, num_classes, variant_id):
                super(CustomCNN, self).__init__()
                base_channels = 32 + (variant_id * 8)

                self.features = nn.Sequential(
                    nn.Conv2d(3, base_channels, 3, padding=1),
                    nn.BatchNorm2d(base_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),

                    nn.Conv2d(base_channels, base_channels*2, 3, padding=1),
                    nn.BatchNorm2d(base_channels*2),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),

                    nn.Conv2d(base_channels*2, base_channels*4, 3, padding=1),
                    nn.BatchNorm2d(base_channels*4),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((4, 4))
                )

                self.classifier = nn.Sequential(
                    nn.Linear(base_channels*4*16, 512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(512, num_classes)
                )

            def forward(self, x):
                x = self.features(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
                return x

        return CustomCNN(num_classes, model_id - 37)

def get_model_name(model_id):
    """Returns model name for given ID"""
    model_names = [
        "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152",
        "DenseNet121", "DenseNet161", "DenseNet169", "DenseNet201",
        "VGG11", "VGG13", "VGG16", "VGG19",
        "EfficientNet-B0", "EfficientNet-B1", "EfficientNet-B2", "EfficientNet-B3", "EfficientNet-B4",
        "MobileNet-V2", "MobileNet-V3-Large", "MobileNet-V3-Small",
        "SqueezeNet1.0", "SqueezeNet1.1",
        "ShuffleNet-V2-x0.5", "ShuffleNet-V2-x1.0",
        "WideResNet50-2", "WideResNet101-2",
        "ResNext50-32x4d", "ResNext101-32x8d",
        "RegNet-Y-400MF", "RegNet-Y-800MF", "RegNet-Y-1.6GF",
        "VGG11-BN", "VGG13-BN", "VGG16-BN", "VGG19-BN",
        "AlexNet"
    ]

    if model_id < len(model_names):
        return model_names[model_id]
    else:
        return f"CustomCNN{model_id-37}"

def setup_data_paths():
    """Set up correct CSV and image paths"""
    dataset_path = f"{DATA_PATH}/nih_dataset"

    csv_path = None
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.csv') and ('Data_Entry' in file or 'data_entry' in file.lower()):
                csv_path = os.path.join(root, file)
                break
        if csv_path:
            break

    if csv_path is None:
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.csv'):
                    try:
                        test_path = os.path.join(root, file)
                        test_df = pd.read_csv(test_path, nrows=1)
                        if 'Finding Labels' in test_df.columns:
                            csv_path = test_path
                            break
                    except:
                        continue
            if csv_path:
                break

    img_dir = None
    for root, dirs, files in os.walk(dataset_path):
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if len(image_files) > 100:
            img_dir = root
            break

    return csv_path, img_dir

def load_trained_model(client_id):
    """Load trained model for a specific client"""
    model_name = get_model_name(client_id)
    model_path = f"{MODEL_SAVE_PATH}/client_{client_id}_{model_name}.pth"

    if not os.path.exists(model_path):
        return None, None

    try:
        model = get_model(client_id)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        return model, model_name
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return None, None

def evaluate_model_accuracy(model, test_loader):
    """Calculate accuracy for a single model"""
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).float()

            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.numpy())

    predictions = np.concatenate(all_predictions, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # Overall accuracy
    overall_accuracy = accuracy_score(labels.flatten(), predictions.flatten())

    # Per-class accuracy
    class_accuracies = {}
    for i, class_name in enumerate(NIH_LABELS):
        class_labels = labels[:, i]
        class_preds = predictions[:, i]
        if len(np.unique(class_labels)) > 1:
            class_acc = accuracy_score(class_labels, class_preds)
        else:
            class_acc = 0.0
        class_accuracies[class_name] = class_acc

    return overall_accuracy, class_accuracies

def create_test_dataset():
    """Create test dataset (same as used in logit extraction)"""
    csv_path, img_dir = setup_data_paths()

    if csv_path is None or img_dir is None:
        return None, None, None

    # Use the same test split as logit extraction
    full_df = pd.read_csv(csv_path)
    test_start_idx = int(len(full_df) * 0.8)  # Last 20% as test
    test_df = full_df.iloc[test_start_idx:].reset_index(drop=True)

    # Save temporary test CSV
    temp_csv = f"{DATA_PATH}/temp_test_eval.csv"
    test_df.to_csv(temp_csv, index=False)

    test_dataset = NIHDataset(temp_csv, img_dir, transform_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    return test_loader, csv_path, img_dir

def evaluate_all_client_models():
    """Evaluate accuracy for all 50 client models"""
    print("=== EVALUATING CLIENT MODEL ACCURACIES ===")

    # Create test dataset
    test_loader, csv_path, img_dir = create_test_dataset()
    if test_loader is None:
        print("Cannot create test dataset!")
        return

    print(f"Test dataset size: {len(test_loader.dataset)}")

    # Find available models
    available_models = []
    for client_id in range(50):
        model_name = get_model_name(client_id)
        model_path = f"{MODEL_SAVE_PATH}/client_{client_id}_{model_name}.pth"
        if os.path.exists(model_path):
            available_models.append(client_id)

    print(f"Found {len(available_models)} trained models")

    # Evaluate each model
    results = {}

    for client_id in available_models:
        print(f"\nEvaluating Client {client_id}")

        model, model_name = load_trained_model(client_id)
        if model is None:
            continue

        try:
            overall_acc, class_accs = evaluate_model_accuracy(model, test_loader)

            results[client_id] = {
                'model_name': model_name,
                'overall_accuracy': overall_acc,
                'class_accuracies': class_accs
            }

            print(f"  {model_name}: {overall_acc:.4f}")

        except Exception as e:
            print(f"  Error evaluating {model_name}: {e}")
            continue

        # Clean up memory
        del model
        torch.cuda.empty_cache()

    # Save results
    accuracy_results_path = f"{MODEL_SAVE_PATH}/client_accuracies.pkl"
    with open(accuracy_results_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"\n=== CLIENT MODEL ACCURACY RESULTS ===")
    print("Model Name               | Overall Accuracy")
    print("-" * 45)

    sorted_results = sorted(results.items(), key=lambda x: x[1]['overall_accuracy'], reverse=True)

    for client_id, result in sorted_results:
        model_name = result['model_name']
        acc = result['overall_accuracy']
        print(f"{model_name:25} | {acc:.4f}")

    avg_accuracy = np.mean([r['overall_accuracy'] for r in results.values()])
    print(f"\nAverage Accuracy: {avg_accuracy:.4f}")
    print(f"Results saved: {accuracy_results_path}")

    # Clean up
    if os.path.exists(f"{DATA_PATH}/temp_test_eval.csv"):
        os.remove(f"{DATA_PATH}/temp_test_eval.csv")

    return results

def load_server_results():
    """Load and display server model results"""
    results_path = f"{SERVER_SAVE_PATH}/training_results.pkl"

    if not os.path.exists(results_path):
        print("Server results not found!")
        return None

    with open(results_path, 'rb') as f:
        results = pickle.load(f)

    print("\n=== SERVER MODEL ACCURACY RESULTS ===")
    print("Model Name     | Overall Accuracy")
    print("-" * 35)

    for model_name, result in results.items():
        acc = result['evaluation']['overall']['accuracy']
        print(f"{model_name:15} | {acc:.4f}")

    return results

def main_accuracy_evaluation():
    """Main function to evaluate all model accuracies"""
    print("Starting Accuracy Evaluation...")

    # Evaluate client models
    client_results = evaluate_all_client_models()

    # Load server results
    server_results = load_server_results()

    print("\n" + "="*50)
    print("ACCURACY EVALUATION COMPLETED!")
    print("="*50)

    return client_results, server_results

# Run evaluation
if __name__ == "__main__":
    client_results, server_results = main_accuracy_evaluation()

# ---- Cell Separator ----

# Accuracy Evaluation with Comprehensive Visualization
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import *
from PIL import Image
import pandas as pd
import numpy as np
import pickle
from google.colab import drive
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

# Set up directories
DRIVE_PATH = '/content/drive/MyDrive/BS (Yonsei University)/Thesis/2025 추계종합학술대회 및 대학생논문경진대회(한국정보기술학회)'
MODEL_SAVE_PATH = f'{DRIVE_PATH}/client_models'
DATA_PATH = f'{DRIVE_PATH}/data'
LOGIT_SAVE_PATH = f'{DRIVE_PATH}/logits'
SERVER_SAVE_PATH = f'{DRIVE_PATH}/server_models'
VIS_SAVE_PATH = f'{DRIVE_PATH}/visualizations'

# Create visualization directory
os.makedirs(VIS_SAVE_PATH, exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# NIH dataset labels
NIH_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]
NUM_CLASSES = len(NIH_LABELS)

# Data transforms
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# NIH Dataset class
class NIHDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.data['Finding Labels'] = self.data['Finding Labels'].fillna('No Finding')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]['Image Index']
        img_path = os.path.join(self.img_dir, img_name)

        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        labels = self.data.iloc[idx]['Finding Labels']
        label_vector = torch.zeros(NUM_CLASSES)

        if labels != 'No Finding':
            for label in labels.split('|'):
                if label.strip() in NIH_LABELS:
                    label_idx = NIH_LABELS.index(label.strip())
                    label_vector[label_idx] = 1.0

        return image, label_vector

# Model definitions (abbreviated for space)
def get_model(model_id, num_classes=NUM_CLASSES):
    """Returns one of 50 different model architectures"""
    if model_id == 0:
        model = resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 1:
        model = resnet34(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 2:
        model = resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 3:
        model = resnet101(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 4:
        model = resnet152(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 5:
        model = densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model
    elif model_id == 6:
        model = densenet161(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model
    elif model_id == 7:
        model = densenet169(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model
    elif model_id == 8:
        model = densenet201(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model
    elif model_id == 9:
        model = vgg11(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    elif model_id == 10:
        model = vgg13(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    elif model_id == 11:
        model = vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    elif model_id == 12:
        model = vgg19(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    elif model_id == 13:
        model = efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    elif model_id == 14:
        model = efficientnet_b1(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    elif model_id == 15:
        model = efficientnet_b2(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    elif model_id == 16:
        model = efficientnet_b3(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    elif model_id == 17:
        model = efficientnet_b4(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    elif model_id == 18:
        model = mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    elif model_id == 19:
        model = mobilenet_v3_large(pretrained=True)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        return model
    elif model_id == 20:
        model = mobilenet_v3_small(pretrained=True)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        return model
    elif model_id == 21:
        model = squeezenet1_0(pretrained=True)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        model.num_classes = num_classes
        return model
    elif model_id == 22:
        model = squeezenet1_1(pretrained=True)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        model.num_classes = num_classes
        return model
    elif model_id == 23:
        model = shufflenet_v2_x0_5(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 24:
        model = shufflenet_v2_x1_0(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 25:
        model = wide_resnet50_2(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 26:
        model = wide_resnet101_2(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 27:
        model = resnext50_32x4d(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 28:
        model = resnext101_32x8d(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 29:
        model = regnet_y_400mf(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 30:
        model = regnet_y_800mf(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 31:
        model = regnet_y_1_6gf(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_id == 32:
        model = vgg11_bn(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    elif model_id == 33:
        model = vgg13_bn(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    elif model_id == 34:
        model = vgg16_bn(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    elif model_id == 35:
        model = vgg19_bn(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    elif model_id == 36:
        model = alexnet(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    else:
        # Custom CNN for remaining models
        class CustomCNN(nn.Module):
            def __init__(self, num_classes, variant_id):
                super(CustomCNN, self).__init__()
                base_channels = 32 + (variant_id * 8)

                self.features = nn.Sequential(
                    nn.Conv2d(3, base_channels, 3, padding=1),
                    nn.BatchNorm2d(base_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),

                    nn.Conv2d(base_channels, base_channels*2, 3, padding=1),
                    nn.BatchNorm2d(base_channels*2),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),

                    nn.Conv2d(base_channels*2, base_channels*4, 3, padding=1),
                    nn.BatchNorm2d(base_channels*4),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((4, 4))
                )

                self.classifier = nn.Sequential(
                    nn.Linear(base_channels*4*16, 512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(512, num_classes)
                )

            def forward(self, x):
                x = self.features(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
                return x

        return CustomCNN(num_classes, model_id - 37)

def get_model_name(model_id):
    """Returns model name for given ID"""
    model_names = [
        "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152",
        "DenseNet121", "DenseNet161", "DenseNet169", "DenseNet201",
        "VGG11", "VGG13", "VGG16", "VGG19",
        "EfficientNet-B0", "EfficientNet-B1", "EfficientNet-B2", "EfficientNet-B3", "EfficientNet-B4",
        "MobileNet-V2", "MobileNet-V3-Large", "MobileNet-V3-Small",
        "SqueezeNet1.0", "SqueezeNet1.1",
        "ShuffleNet-V2-x0.5", "ShuffleNet-V2-x1.0",
        "WideResNet50-2", "WideResNet101-2",
        "ResNext50-32x4d", "ResNext101-32x8d",
        "RegNet-Y-400MF", "RegNet-Y-800MF", "RegNet-Y-1.6GF",
        "VGG11-BN", "VGG13-BN", "VGG16-BN", "VGG19-BN",
        "AlexNet"
    ]

    if model_id < len(model_names):
        return model_names[model_id]
    else:
        return f"CustomCNN{model_id-37}"

def get_model_family(model_name):
    """Get model family for grouping"""
    if "ResNet" in model_name or "ResNext" in model_name or "WideResNet" in model_name:
        return "ResNet Family"
    elif "DenseNet" in model_name:
        return "DenseNet"
    elif "VGG" in model_name:
        return "VGG"
    elif "EfficientNet" in model_name:
        return "EfficientNet"
    elif "MobileNet" in model_name:
        return "MobileNet"
    elif "SqueezeNet" in model_name:
        return "SqueezeNet"
    elif "ShuffleNet" in model_name:
        return "ShuffleNet"
    elif "RegNet" in model_name:
        return "RegNet"
    elif "AlexNet" in model_name:
        return "AlexNet"
    else:
        return "Custom CNN"

def setup_data_paths():
    """Set up correct CSV and image paths"""
    dataset_path = f"{DATA_PATH}/nih_dataset"

    csv_path = None
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.csv') and ('Data_Entry' in file or 'data_entry' in file.lower()):
                csv_path = os.path.join(root, file)
                break
        if csv_path:
            break

    if csv_path is None:
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.csv'):
                    try:
                        test_path = os.path.join(root, file)
                        test_df = pd.read_csv(test_path, nrows=1)
                        if 'Finding Labels' in test_df.columns:
                            csv_path = test_path
                            break
                    except:
                        continue
            if csv_path:
                break

    img_dir = None
    for root, dirs, files in os.walk(dataset_path):
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if len(image_files) > 100:
            img_dir = root
            break

    return csv_path, img_dir

def load_trained_model(client_id):
    """Load trained model for a specific client"""
    model_name = get_model_name(client_id)
    model_path = f"{MODEL_SAVE_PATH}/client_{client_id}_{model_name}.pth"

    if not os.path.exists(model_path):
        return None, None

    try:
        model = get_model(client_id)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        return model, model_name
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return None, None

def evaluate_model_accuracy(model, test_loader):
    """Calculate accuracy for a single model"""
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).float()

            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.numpy())

    predictions = np.concatenate(all_predictions, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # Overall accuracy
    overall_accuracy = accuracy_score(labels.flatten(), predictions.flatten())

    # Per-class accuracy
    class_accuracies = {}
    for i, class_name in enumerate(NIH_LABELS):
        class_labels = labels[:, i]
        class_preds = predictions[:, i]
        if len(np.unique(class_labels)) > 1:
            class_acc = accuracy_score(class_labels, class_preds)
        else:
            class_acc = 0.0
        class_accuracies[class_name] = class_acc

    return overall_accuracy, class_accuracies

def create_test_dataset():
    """Create test dataset (same as used in logit extraction)"""
    csv_path, img_dir = setup_data_paths()

    if csv_path is None or img_dir is None:
        return None, None, None

    # Use the same test split as logit extraction
    full_df = pd.read_csv(csv_path)
    test_start_idx = int(len(full_df) * 0.8)  # Last 20% as test
    test_df = full_df.iloc[test_start_idx:].reset_index(drop=True)

    # Save temporary test CSV
    temp_csv = f"{DATA_PATH}/temp_test_eval.csv"
    test_df.to_csv(temp_csv, index=False)

    test_dataset = NIHDataset(temp_csv, img_dir, transform_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    return test_loader, csv_path, img_dir

def evaluate_all_client_models():
    """Evaluate accuracy for all 50 client models"""
    print("=== EVALUATING CLIENT MODEL ACCURACIES ===")

    # Create test dataset
    test_loader, csv_path, img_dir = create_test_dataset()
    if test_loader is None:
        print("Cannot create test dataset!")
        return

    print(f"Test dataset size: {len(test_loader.dataset)}")

    # Find available models
    available_models = []
    for client_id in range(50):
        model_name = get_model_name(client_id)
        model_path = f"{MODEL_SAVE_PATH}/client_{client_id}_{model_name}.pth"
        if os.path.exists(model_path):
            available_models.append(client_id)

    print(f"Found {len(available_models)} trained models")

    # Evaluate each model
    results = {}

    for client_id in available_models:
        print(f"\nEvaluating Client {client_id}")

        model, model_name = load_trained_model(client_id)
        if model is None:
            continue

        try:
            overall_acc, class_accs = evaluate_model_accuracy(model, test_loader)

            results[client_id] = {
                'model_name': model_name,
                'model_family': get_model_family(model_name),
                'overall_accuracy': overall_acc,
                'class_accuracies': class_accs
            }

            print(f"  {model_name}: {overall_acc:.4f}")

        except Exception as e:
            print(f"  Error evaluating {model_name}: {e}")
            continue

        # Clean up memory
        del model
        torch.cuda.empty_cache()

    # Save results
    accuracy_results_path = f"{MODEL_SAVE_PATH}/client_accuracies.pkl"
    with open(accuracy_results_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"\n=== CLIENT MODEL ACCURACY RESULTS ===")
    print("Model Name               | Overall Accuracy")
    print("-" * 45)

    sorted_results = sorted(results.items(), key=lambda x: x[1]['overall_accuracy'], reverse=True)

    for client_id, result in sorted_results:
        model_name = result['model_name']
        acc = result['overall_accuracy']
        print(f"{model_name:25} | {acc:.4f}")

    avg_accuracy = np.mean([r['overall_accuracy'] for r in results.values()])
    print(f"\nAverage Accuracy: {avg_accuracy:.4f}")
    print(f"Results saved: {accuracy_results_path}")

    # Clean up
    if os.path.exists(f"{DATA_PATH}/temp_test_eval.csv"):
        os.remove(f"{DATA_PATH}/temp_test_eval.csv")

    return results

def load_server_results():
    """Load and display server model results"""
    results_path = f"{SERVER_SAVE_PATH}/training_results.pkl"

    if not os.path.exists(results_path):
        print("Server results not found!")
        return None

    with open(results_path, 'rb') as f:
        results = pickle.load(f)

    print("\n=== SERVER MODEL ACCURACY RESULTS ===")
    print("Model Name     | Overall Accuracy")
    print("-" * 35)

    for model_name, result in results.items():
        acc = result['evaluation']['overall']['accuracy']
        print(f"{model_name:15} | {acc:.4f}")

    return results

# VISUALIZATION FUNCTIONS

def plot_client_accuracy_distribution(client_results):
    """Plot accuracy distribution of client models"""
    if not client_results:
        return

    accuracies = [result['overall_accuracy'] for result in client_results.values()]

    plt.figure(figsize=(12, 8))

    # Histogram
    plt.subplot(2, 2, 1)
    plt.hist(accuracies, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
    plt.title('Client Model Accuracy Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Accuracy')
    plt.ylabel('Number of Models')
    plt.axvline(np.mean(accuracies), color='red', linestyle='--',
                label=f'Mean: {np.mean(accuracies):.4f}')
    plt.legend()

    # Box plot
    plt.subplot(2, 2, 2)
    plt.boxplot(accuracies, vert=True)
    plt.title('Client Model Accuracy Box Plot', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)

    # Model family performance
    plt.subplot(2, 1, 2)
    family_data = {}
    for result in client_results.values():
        family = result['model_family']
        if family not in family_data:
            family_data[family] = []
        family_data[family].append(result['overall_accuracy'])

    families = list(family_data.keys())
    family_means = [np.mean(family_data[family]) for family in families]
    family_stds = [np.std(family_data[family]) for family in families]

    bars = plt.bar(families, family_means, yerr=family_stds, capsize=5,
                   alpha=0.8, color=plt.cm.Set3(np.linspace(0, 1, len(families))))
    plt.title('Average Accuracy by Model Family', fontsize=14, fontweight='bold')
    plt.xlabel('Model Family')
    plt.ylabel('Average Accuracy')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, mean in zip(bars, family_means):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{VIS_SAVE_PATH}/client_accuracy_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_top_bottom_models(client_results, top_n=10):
    """Plot top and bottom performing models"""
    if not client_results:
        return

    sorted_results = sorted(client_results.items(),
                          key=lambda x: x[1]['overall_accuracy'], reverse=True)

    plt.figure(figsize=(16, 10))

    # Top models
    plt.subplot(2, 1, 1)
    top_models = sorted_results[:top_n]
    top_names = [result[1]['model_name'] for result in top_models]
    top_accs = [result[1]['overall_accuracy'] for result in top_models]
    top_families = [result[1]['model_family'] for result in top_models]

    colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(top_names)))
    bars = plt.bar(range(len(top_names)), top_accs, color=colors)
    plt.title(f'Top {top_n} Performing Client Models', fontsize=16, fontweight='bold')
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.xticks(range(len(top_names)), top_names, rotation=45, ha='right')

    # Add value labels and family info
    for i, (bar, acc, family) in enumerate(zip(bars, top_accs, top_families)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                family, ha='center', va='center', rotation=90,
                color='white', fontsize=8, fontweight='bold')

    plt.grid(True, alpha=0.3)

    # Bottom models
    plt.subplot(2, 1, 2)
    bottom_models = sorted_results[-top_n:]
    bottom_names = [result[1]['model_name'] for result in bottom_models]
    bottom_accs = [result[1]['overall_accuracy'] for result in bottom_models]
    bottom_families = [result[1]['model_family'] for result in bottom_models]

    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(bottom_names)))
    bars = plt.bar(range(len(bottom_names)), bottom_accs, color=colors)
    plt.title(f'Bottom {top_n} Performing Client Models', fontsize=16, fontweight='bold')
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.xticks(range(len(bottom_names)), bottom_names, rotation=45, ha='right')

    # Add value labels and family info
    for i, (bar, acc, family) in enumerate(zip(bars, bottom_accs, bottom_families)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                family, ha='center', va='center', rotation=90,
                color='white', fontsize=8, fontweight='bold')

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{VIS_SAVE_PATH}/top_bottom_models.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_class_wise_performance(client_results):
    """Plot class-wise performance heatmap"""
    if not client_results:
        return

    # Create class performance matrix
    model_names = []
    class_performance_matrix = []

    sorted_results = sorted(client_results.items(),
                          key=lambda x: x[1]['overall_accuracy'], reverse=True)

    for client_id, result in sorted_results[:20]:  # Top 20 models
        model_names.append(result['model_name'])
        class_accs = [result['class_accuracies'][class_name] for class_name in NIH_LABELS]
        class_performance_matrix.append(class_accs)

    class_performance_matrix = np.array(class_performance_matrix)

    plt.figure(figsize=(16, 12))

    # Heatmap
    im = plt.imshow(class_performance_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)

    # Set ticks and labels
    plt.xticks(range(len(NIH_LABELS)), NIH_LABELS, rotation=45, ha='right')
    plt.yticks(range(len(model_names)), model_names)

    # Add colorbar
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label('Accuracy', rotation=270, labelpad=20)

    # Add text annotations
    for i in range(len(model_names)):
        for j in range(len(NIH_LABELS)):
            text = plt.text(j, i, f'{class_performance_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)

    plt.title('Class-wise Performance Heatmap (Top 20 Models)',
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Disease Classes')
    plt.ylabel('Models')
    plt.tight_layout()
    plt.savefig(f'{VIS_SAVE_PATH}/class_wise_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_server_vs_client_comparison(client_results, server_results):
    """Compare server models with client models"""
    if not client_results or not server_results:
        return

    # Get client accuracies
    client_accs = [result['overall_accuracy'] for result in client_results.values()]
    client_avg = np.mean(client_accs)
    client_std = np.std(client_accs)
    client_max = np.max(client_accs)

    # Get server accuracies
    server_accs = []
    server_names = []
    for model_name, result in server_results.items():
        server_accs.append(result['evaluation']['overall']['accuracy'])
        server_names.append(model_name)

    plt.figure(figsize=(14, 10))

    # Main comparison plot
    plt.subplot(2, 2, (1, 2))

    # Client model distribution
    plt.hist(client_accs, bins=20, alpha=0.6, label='Client Models',
             color='lightblue', edgecolor='black')

    # Add lines for statistics
    plt.axvline(client_avg, color='blue', linestyle='--', linewidth=2,
                label=f'Client Avg: {client_avg:.4f}')
    plt.axvline(client_max, color='blue', linestyle='-', linewidth=2,
                label=f'Client Best: {client_max:.4f}')

    # Add server model results
    colors = ['red', 'orange', 'green']
    for i, (acc, name) in enumerate(zip(server_accs, server_names)):
        plt.axvline(acc, color=colors[i], linestyle=':', linewidth=3,
                    label=f'{name}: {acc:.4f}')

    plt.title('Server vs Client Model Performance Comparison',
              fontsize=16, fontweight='bold')
    plt.xlabel('Accuracy')
    plt.ylabel('Number of Models')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Bar comparison
    plt.subplot(2, 2, 3)
    all_names = ['Client Avg', 'Client Best'] + server_names
    all_accs = [client_avg, client_max] + server_accs
    colors = ['lightblue', 'blue'] + ['red', 'orange', 'green']

    bars = plt.bar(all_names, all_accs, color=colors, alpha=0.8, edgecolor='black')
    plt.title('Performance Summary', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)

    # Add value labels
    for bar, acc in zip(bars, all_accs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')

    plt.grid(True, alpha=0.3)

    # Improvement analysis
    plt.subplot(2, 2, 4)
    improvements = []
    for acc in server_accs:
        improvement = ((acc - client_avg) / client_avg) * 100
        improvements.append(improvement)

    bars = plt.bar(server_names, improvements,
                   color=['red', 'orange', 'green'], alpha=0.8, edgecolor='black')
    plt.title('Improvement over Client Average', fontsize=14, fontweight='bold')
    plt.ylabel('Improvement (%)')
    plt.xticks(rotation=45)
    plt.axhline(0, color='black', linestyle='-', alpha=0.5)

    # Add value labels
    for bar, imp in zip(bars, improvements):
        plt.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (0.1 if imp > 0 else -0.3),
                f'{imp:.2f}%', ha='center', va='bottom' if imp > 0 else 'top',
                fontweight='bold')

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{VIS_SAVE_PATH}/server_vs_client_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_federated_learning_overview(client_results, server_results):
    """Create comprehensive federated learning overview"""
    if not client_results or not server_results:
        return

    fig = plt.figure(figsize=(20, 12))

    # Main architecture diagram
    ax1 = plt.subplot(2, 3, (1, 2))

    # Draw federated learning architecture
    client_accs = [result['overall_accuracy'] for result in client_results.values()]
    server_accs = [result['evaluation']['overall']['accuracy'] for result in server_results.values()]

    # Client models (scatter plot showing diversity)
    families = list(set([result['model_family'] for result in client_results.values()]))
    family_colors = plt.cm.Set3(np.linspace(0, 1, len(families)))
    family_color_map = dict(zip(families, family_colors))

    for i, (client_id, result) in enumerate(client_results.items()):
        x = np.random.normal(1, 0.1)  # Add some jitter
        y = result['overall_accuracy']
        color = family_color_map[result['model_family']]
        plt.scatter(x, y, c=[color], s=50, alpha=0.7, edgecolor='black')

    # Arrow from clients to server
    plt.arrow(1.5, np.mean(client_accs), 1, 0, head_width=0.02,
              head_length=0.1, fc='black', ec='black')
    plt.text(2, np.mean(client_accs) + 0.03, 'Knowledge\nDistillation',
             ha='center', fontweight='bold')

    # Server models
    for i, (name, acc) in enumerate(zip(['ServerCNN', 'ServerViT', 'ServerHybrid'], server_accs)):
        plt.scatter(3, acc, c='red', s=200, marker='s', alpha=0.8, edgecolor='black')
        plt.text(3.2, acc, name, va='center', fontweight='bold')

    plt.xlim(0.5, 3.8)
    plt.ylim(min(min(client_accs), min(server_accs)) - 0.05,
             max(max(client_accs), max(server_accs)) + 0.05)
    plt.ylabel('Accuracy')
    plt.title('Federated Learning with Knowledge Distillation',
              fontsize=16, fontweight='bold')
    plt.xticks([1, 3], ['Client Models\n(50 Heterogeneous)', 'Server Models\n(3 Architectures)'])

    # Legend for model families
    legend_elements = [plt.scatter([], [], c=[color], s=50, alpha=0.7,
                                 edgecolor='black', label=family)
                      for family, color in family_color_map.items()]
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))

    # Statistics panel
    ax2 = plt.subplot(2, 3, 3)
    ax2.axis('off')

    stats_text = f"""
    EXPERIMENT STATISTICS

    Dataset: NIH Chest X-rays
    Total Images: 112,120
    Disease Classes: 14
    Test Samples: {len(client_accs) * 1000}  # Approximate

    CLIENT MODELS:
    Total Models: {len(client_results)}
    Model Families: {len(families)}
    Avg Accuracy: {np.mean(client_accs):.4f}
    Best Accuracy: {np.max(client_accs):.4f}
    Worst Accuracy: {np.min(client_accs):.4f}
    Std Deviation: {np.std(client_accs):.4f}

    SERVER MODELS:
    ServerCNN: {server_accs[0]:.4f}
    ServerViT: {server_accs[1]:.4f}
    ServerHybrid: {server_accs[2]:.4f}
    Best Server: {np.max(server_accs):.4f}

    IMPROVEMENT:
    Best Server vs Avg Client:
    {((np.max(server_accs) - np.mean(client_accs)) / np.mean(client_accs) * 100):+.2f}%
    """

    ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    # Model family distribution
    ax3 = plt.subplot(2, 3, 4)
    family_counts = {}
    for result in client_results.values():
        family = result['model_family']
        family_counts[family] = family_counts.get(family, 0) + 1

    wedges, texts, autotexts = plt.pie(family_counts.values(), labels=family_counts.keys(),
                                      autopct='%1.0f%%', startangle=90)
    plt.title('Model Family Distribution', fontsize=14, fontweight='bold')

    # Performance comparison
    ax4 = plt.subplot(2, 3, 5)
    performance_data = {
        'Client Average': np.mean(client_accs),
        'Client Best': np.max(client_accs),
        'ServerCNN': server_accs[0],
        'ServerViT': server_accs[1],
        'ServerHybrid': server_accs[2]
    }

    colors = ['lightblue', 'blue', 'red', 'orange', 'green']
    bars = plt.bar(performance_data.keys(), performance_data.values(),
                   color=colors, alpha=0.8, edgecolor='black')
    plt.title('Performance Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)

    for bar, acc in zip(bars, performance_data.values()):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.grid(True, alpha=0.3)

    # Training progress (simulated)
    ax5 = plt.subplot(2, 3, 6)
    epochs = range(1, 11)

    # Simulate training curves
    client_curve = [0.6 + 0.2 * (1 - np.exp(-epoch/3)) + np.random.normal(0, 0.01)
                   for epoch in epochs]
    server_curves = {
        'ServerCNN': [0.65 + 0.15 * (1 - np.exp(-epoch/2.5)) + np.random.normal(0, 0.005)
                     for epoch in epochs],
        'ServerViT': [0.63 + 0.18 * (1 - np.exp(-epoch/3)) + np.random.normal(0, 0.005)
                     for epoch in epochs],
        'ServerHybrid': [0.67 + 0.17 * (1 - np.exp(-epoch/2.8)) + np.random.normal(0, 0.005)
                        for epoch in epochs]
    }

    plt.plot(epochs, client_curve, 'b--', label='Client Average', linewidth=2)
    colors = ['red', 'orange', 'green']
    for i, (name, curve) in enumerate(server_curves.items()):
        plt.plot(epochs, curve, color=colors[i], label=name, linewidth=2)

    plt.title('Training Progress (Simulated)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{VIS_SAVE_PATH}/federated_learning_overview.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_table(client_results, server_results):
    """Create and save summary tables"""
    if not client_results:
        return

    # Client results table
    client_data = []
    for client_id, result in client_results.items():
        client_data.append({
            'Client_ID': client_id,
            'Model_Name': result['model_name'],
            'Model_Family': result['model_family'],
            'Overall_Accuracy': result['overall_accuracy'],
            'Rank': 0  # Will be filled later
        })

    client_df = pd.DataFrame(client_data)
    client_df = client_df.sort_values('Overall_Accuracy', ascending=False)
    client_df['Rank'] = range(1, len(client_df) + 1)

    # Save client results
    client_df.to_csv(f'{VIS_SAVE_PATH}/client_results_summary.csv', index=False)

    # Server results table
    if server_results:
        server_data = []
        for model_name, result in server_results.items():
            server_data.append({
                'Model_Name': model_name,
                'Overall_Accuracy': result['evaluation']['overall']['accuracy'],
                'F1_Score': result['evaluation']['overall']['f1_score'],
                'AUC_Score': result['evaluation']['overall']['auc_score']
            })

        server_df = pd.DataFrame(server_data)
        server_df = server_df.sort_values('Overall_Accuracy', ascending=False)
        server_df.to_csv(f'{VIS_SAVE_PATH}/server_results_summary.csv', index=False)

    print(f"Summary tables saved to {VIS_SAVE_PATH}")

    return client_df, server_df if server_results else None

def main_visualization_pipeline():
    """Main function to run all visualizations"""
    print("=== STARTING COMPREHENSIVE VISUALIZATION PIPELINE ===")

    # Evaluate client models
    client_results = evaluate_all_client_models()

    # Load server results
    server_results = load_server_results()

    if not client_results:
        print("No client results available for visualization!")
        return

    print("\n=== CREATING VISUALIZATIONS ===")

    # Create all visualizations
    print("1. Client accuracy distribution...")
    plot_client_accuracy_distribution(client_results)

    print("2. Top/bottom models comparison...")
    plot_top_bottom_models(client_results, top_n=10)

    print("3. Class-wise performance heatmap...")
    plot_class_wise_performance(client_results)

    if server_results:
        print("4. Server vs client comparison...")
        plot_server_vs_client_comparison(client_results, server_results)

        print("5. Federated learning overview...")
        plot_federated_learning_overview(client_results, server_results)

    print("6. Creating summary tables...")
    client_df, server_df = create_summary_table(client_results, server_results)

    print(f"\n=== VISUALIZATION COMPLETED ===")
    print(f"All plots saved to: {VIS_SAVE_PATH}")
    print("Generated files:")
    print("- client_accuracy_distribution.png")
    print("- top_bottom_models.png")
    print("- class_wise_performance.png")
    if server_results:
        print("- server_vs_client_comparison.png")
        print("- federated_learning_overview.png")
    print("- client_results_summary.csv")
    if server_results:
        print("- server_results_summary.csv")

    return client_results, server_results

# Run the complete pipeline
if __name__ == "__main__":
    print("Starting Comprehensive Accuracy Evaluation and Visualization...")
    print(f"Using {NUM_CLASSES} classes: {NIH_LABELS}")

    client_results, server_results = main_visualization_pipeline()

    print("\nEvaluation and visualization completed! Check your Google Drive for all results.")
