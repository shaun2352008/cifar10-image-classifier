"""Small CNN image classifier on CIFAR-10 (PyTorch).

Trains for a few epochs, prints test accuracy each epoch,
and saves a confusion matrix image at the end.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.001

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 1. Data: CIFAR-10 downloads automatically on first run
transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
train_set = torchvision.datasets.CIFAR10("data", train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10("data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)
classes = train_set.classes


# 2. Model: two conv blocks + a small classifier head
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 32x32 -> 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16x16 -> 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model = SmallCNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# 3. Evaluation helper
def evaluate():
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for images, y in test_loader:
            out = model(images.to(device))
            preds.extend(out.argmax(1).cpu().tolist())
            labels.extend(y.tolist())
    acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)
    return acc, preds, labels


# 4. Training loop
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for images, y in train_loader:
        images, y = images.to(device), y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(images), y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    acc, _, _ = evaluate()
    print(f"Epoch {epoch}/{EPOCHS} | loss {running_loss / len(train_loader):.3f} | test accuracy {acc:.1%}")

# 5. Final accuracy + confusion matrix
acc, preds, labels = evaluate()
print(f"\nFinal test accuracy: {acc:.1%}")

cm = confusion_matrix(labels, preds)
fig, ax = plt.subplots(figsize=(8, 8))
ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, xticks_rotation=45, colorbar=False)
plt.title(f"CIFAR-10 confusion matrix (accuracy {acc:.1%})")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("Saved confusion_matrix.png")
