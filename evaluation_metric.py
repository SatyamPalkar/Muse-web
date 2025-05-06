import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# === Load Dataset ===
data = np.load("eeg_emotion_balanced_dataset.npz")
X = data['X'].astype(np.float32)
y = data['y'].astype(np.int64)

# === Normalize and reshape ===
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = X[:, np.newaxis, :]  # shape: (samples, 1, seq_len)

X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y)

# === Dataset ===
dataset = TensorDataset(X_tensor, y_tensor)
train_len = int(0.8 * len(dataset))
val_len = len(dataset) - train_len
train_set, val_set = random_split(dataset, [train_len, val_len])

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64)

# === Model Definition ===
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, 1)

    def forward(self, lstm_output):
        weights = torch.softmax(self.attn(lstm_output), dim=1)
        return torch.sum(weights * lstm_output, dim=1)

class CNN_BiLSTM_Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(16, 32, bidirectional=True, batch_first=True)
        self.attn = Attention(32)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, 4)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        context = self.attn(lstm_out)
        context = self.dropout(context)
        return self.fc(context)

# === Training with Focal Loss ===
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        return torch.mean(focal_loss) if self.reduction == 'mean' else focal_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_BiLSTM_Attn().to(device)
criterion = FocalLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# === Training ===
train_losses, val_losses = [], []
epochs = 30

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    train_losses.append(total_loss / len(train_loader))

    # Validation loss
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            val_loss += loss.item()
    val_losses.append(val_loss / len(val_loader))

# === Plot Loss Curves ===
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Training Loss", linewidth=2)
plt.plot(val_losses, label="Validation Loss", linewidth=2)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve (Focal Loss)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Classification Report & Confusion Matrix ===
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.to(device)
        out = model(xb)
        pred = torch.argmax(out, dim=1).cpu().numpy()
        y_true.extend(yb.numpy())
        y_pred.extend(pred)

report = classification_report(y_true, y_pred, target_names=["Relaxed", "Focused", "Stressed", "Drowsy"], output_dict=True)
conf_matrix = confusion_matrix(y_true, y_pred)

# Plot Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Relaxed", "Focused", "Stressed", "Drowsy"],
            yticklabels=["Relaxed", "Focused", "Stressed", "Drowsy"])
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.show()

report
