

!pip install torch torchvision segmentation-models-pytorch

from google.colab import drive
drive.mount('/content/drive')

import os
dataset_path = '/content/drive/MyDrive/dataset'
os.listdir(dataset_path)

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

# Custom Dataset class for image and mask pairs
class SegmentationDataset(Dataset):
    def __init__(self, path , transform = None):
        self.image_dir = imagePath = os.path.join(path, 'images')
        self.mask_dir  = os.path.join(path, 'masks')
        self.transform = transform
        self.images = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace(".jpg", ".png"))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Load datasets
train_dataset = SegmentationDataset('/content/drive/MyDrive/dataset/train', transform=transform)
val_dataset = SegmentationDataset('/content/drive/MyDrive/dataset/val', transform=transform)
test_dataset = SegmentationDataset('/content/drive/MyDrive/dataset/test', transform=transform)
# Dataloaders for batching
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
import segmentation_models_pytorch as smp


model = smp.Unet(
    encoder_name="efficientnet-b7",            # Choose ResNet101 encoder
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(model)

import torch.optim as optim
import torch.nn as nn
import torchvision.transforms.functional as F
import torch.nn.functional as F_nn
import torchvision.transforms as T

for param in model.encoder.parameters():
    param.requires_grad = False

for param in model.encoder._blocks[-1].parameters():
    param.requires_grad = True
for param in model.encoder._blocks[-2].parameters():
    param.requires_grad = True


def dice_loss(pred, target, smooth=1.):
    pred = torch.sigmoid(pred)
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) /
                 (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()

class BCEWithDiceLoss(nn.Module):
    def __init__(self):
        super(BCEWithDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss_val = dice_loss(pred, target)
        return bce_loss * 0.4 + dice_loss_val * 1.6

criterion = BCEWithDiceLoss()

optimizer = optim.AdamW(model.parameters(), lr=0.0001)

def augment(image, mask):
    transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(90),
        T.ColorJitter(brightness=0.5, contrast=0.5),
        T.RandomApply([T.GaussianBlur(3)], p=0.5),
        T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    ])

    image = transform(image)
    mask = transform(mask)

    return image, mask

def post_process(output, threshold=0.5):
    output = (output > threshold).float()
    output = F_nn.max_pool2d(output, kernel_size=3, stride=1, padding=1)
    return output

dropout = nn.Dropout(0.2)

def deep_supervision_loss(outputs, masks):
    total_loss = 0
    for output in outputs:
        total_loss += criterion(output, masks)
    return total_loss / len(outputs)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 10
batch_size = 16

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, masks in train_loader:
        images, masks = augment(images, masks)  # Apply augmentations
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)

        if isinstance(outputs, list):
            loss = deep_supervision_loss(outputs, masks)
        else:
            loss = criterion(outputs, masks)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

model.eval()
dice_coeffs = []
ious = []

with torch.no_grad():
    for images, masks in val_loader:
        images = images.to(device)
        masks = masks.to(device)


        outputs = torch.sigmoid(model(images))
        outputs_flip = torch.sigmoid(model(torch.flip(images, dims=[-1])))


        outputs = (outputs + torch.flip(outputs_flip, dims=[-1])) / 2.0

        outputs = post_process(outputs)

        intersection = (outputs * masks).sum((2, 3))
        dice_coeff = (2. * intersection + 1) / (outputs.sum((2, 3)) + masks.sum((2, 3)) + 1)
        dice_coeffs.append(dice_coeff.mean().item())

        union = (outputs + masks).sum((2, 3)) - intersection
        iou = (intersection + 1) / (union + 1)
        ious.append(iou.mean().item())

avg_dice_coeff = sum(dice_coeffs) / len(dice_coeffs)
avg_iou = sum(ious) / len(ious)

print(f"Average Dice Coefficient: {avg_dice_coeff:.4f}")
print(f"Average IoU: {avg_iou:.4f}")
