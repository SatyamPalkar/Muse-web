# === Improved CNN + BiLSTM + Attention Model Training ===
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# === Load & Normalize Dataset ===
data = np.load("eeg_emotion_balanced_dataset.npz")
X = data['X']  # Expected shape: (N, 5)
y = data['y']

# Normalize each feature (band)
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
X = X.astype(np.float32)
X = np.expand_dims(X, axis=1)  # Shape: (N, 1, 5)

y = y.astype(int)
X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y, dtype=torch.long)

# === Dataset Split ===
dataset = TensorDataset(X_tensor, y_tensor)
indices = np.random.permutation(len(dataset))
split = int(0.8 * len(indices))
train_idx, val_idx = indices[:split], indices[split:]
train_set = Subset(dataset, train_idx)
val_set = Subset(dataset, val_idx)

# === Class Balancing ===
class_weights = compute_class_weight("balanced", classes=np.unique(y[train_idx]), y=y[train_idx])
sample_weights = [class_weights[label] for label in y[train_idx]]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(train_set, batch_size=32, sampler=sampler)
val_loader = DataLoader(val_set, batch_size=1)

# === Attention Layer ===
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, 1)

    def forward(self, lstm_out):
        weights = torch.softmax(self.attn(lstm_out), dim=1)
        return torch.sum(weights * lstm_out, dim=1)

# === Model ===
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, 1)

    def forward(self, lstm_output):
        weights = torch.softmax(self.attn(lstm_output), dim=1)
        context = torch.sum(weights * lstm_output, dim=1)
        return context

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, 1)

    def forward(self, lstm_output):
        weights = torch.softmax(self.attn(lstm_output), dim=1)
        context = torch.sum(weights * lstm_output, dim=1)
        return context

class CNN_BiLSTM_Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),  # Must match training
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, bidirectional=True, batch_first=True)
        self.attn = Attention(64)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, 4)  # 4 output classes
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):  # Input shape: (B, 1, 5)
        x = self.cnn(x)              # (B, 32, 5)
        x = x.permute(0, 2, 1)       # (B, 5, 32)
        lstm_out, _ = self.lstm(x)   # (B, 5, 128)
        context = self.attn(lstm_out)  # (B, 128)
        context = self.dropout(context)
        return self.fc(context) 
# === Training Function ===
def train_model():
    model = CNN_BiLSTM_Attn().to("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(50):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to("cpu"), yb.to("cpu")
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

    # === Evaluation ===
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            out = model(xb)
            pred = torch.argmax(torch.softmax(out, dim=1), dim=1)
            y_true.append(yb.item())
            y_pred.append(pred.item())

    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Relaxed", "Focused", "Stressed", "Drowsy"]))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=["Relaxed", "Focused", "Stressed", "Drowsy"], cmap="Blues")
    plt.title("Confusion Matrix - EEG Emotion Classification")
    plt.show()

    torch.save(model.state_dict(), "emotion_model_new.pth")
    print("✅ Model saved to emotion_model_new.pth")

# === Run Training ===
if __name__ == "__main__":
    train_model()
