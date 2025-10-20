# ====== Initialise ======

import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

base_dir = "/home/jovyan/.cache/kagglehub/datasets/innominate817/hagrid-sample-30k-384p/versions/5/hagrid-sample-30k-384p/hagrid_30k"

# ====== Transforms ======
transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),     # random zooms
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=25),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ====== Dataset & Split ======
dataset = datasets.ImageFolder(base_dir, transform=transform)
num_classes = len(dataset.classes)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4)
val_dl = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4)

print(f"Classes: {dataset.classes}")
print(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")

# ====== Model ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.mobilenet_v3_small(pretrained=True)
for param in model.features.parameters():
    param.requires_grad = True  # fine-tune all layers

# Add dropout + replace classifier head
model.classifier[2] = nn.Dropout(p=0.3)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
model = model.to(device)

# ====== Loss, Optimizer, Scheduler ======
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

# ====== Training Loop ======
epochs = 25
train_losses, val_accuracies = [], []

for epoch in range(epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for imgs, labels in train_dl:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total
    avg_loss = total_loss / len(train_dl)

    # Validation
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
    val_acc = 100 * val_correct / val_total

    train_losses.append(avg_loss)
    val_accuracies.append(val_acc)

    scheduler.step()

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
    
# ====== Save Model ======
torch.save(model.state_dict(), "gesture_model_v2.pth")
print("Saved as gesture_model_v2.pth")

dummy_input = torch.randn(1, 3, 224, 224).to(device)
torch.onnx.export(model, dummy_input, "gesture_model.onnx_v2", opset_version=11)
print("Exported as gesture_model_v2.onnx")

# ====== Plot Training Progress ======
plt.plot(train_losses, label="Training Loss")
plt.plot(val_accuracies, label="Validation Accuracy")
plt.legend(); plt.show()