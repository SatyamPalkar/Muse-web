import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

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
        self.fc = nn.Linear(64, 4)  # 4 output classes

    def forward(self, x):
        x = self.cnn(x)  # (batch, 16, seq_len)
        x = x.permute(0, 2, 1)  # (batch, seq_len, 16)
        lstm_out, _ = self.lstm(x)
        context = self.attn(lstm_out)
        context = self.dropout(context)
        return self.fc(context)

# === Load and Prepare Dataset ===
data = np.load("eeg_emotion_balanced_dataset.npz")
X = data['X'].astype(np.float32)  # shape: (samples, seq_len)
y = data['y'].astype(np.int64)    # shape: (samples,)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = X[:, np.newaxis, :]  # (samples, 1, seq_len)

# Torch tensors
X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y)

# Dataset and loader
dataset = TensorDataset(X_tensor, y_tensor)
train_len = int(0.8 * len(dataset))
train_set, val_set = random_split(dataset, [train_len, len(dataset) - train_len])
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64)

# === Training Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_BiLSTM_Attn().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 50

# === Training Loop ===
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
    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

# === Validation ===
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.to(device)
        out = model(xb)
        preds = torch.argmax(out, dim=1).cpu().numpy()
        y_true.extend(yb.numpy())
        y_pred.extend(preds)
print("\nValidation Report:\n", classification_report(y_true, y_pred, target_names=["Relaxed", "Focused", "Stressed", "Drowsy"]))

# === Save the model ===
torch.save(model.state_dict(), "new_try.pth")
print("✅ Model saved as new_try.pth")
