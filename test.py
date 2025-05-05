import numpy as np
import torch
import torch.nn as nn
from collections import Counter

# === Load and normalize dataset ===
data = np.load("eeg_emotion_balanced_dataset.npz")
X_raw = data['X']
y = data['y']
labels = ["Relaxed", "Focused", "Stressed", "Drowsy"]

def normalize(X):
    return (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)

# === Attention module ===
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, 1)
    def forward(self, lstm_output):
        weights = torch.softmax(self.attn(lstm_output), dim=1)
        return torch.sum(weights * lstm_output, dim=1)

# === CNN (1-layer) + BiLSTM + Attention model (matches trained weights) ===
class CNN_BiLSTM_Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(16, 32, bidirectional=True, batch_first=True)
        self.attn = Attention(32)
        self.fc = nn.Linear(64, 4)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, 1, T)
        x = self.cnn(x)         # (B, 16, T)
        x = x.permute(0, 2, 1)  # (B, T, 16)
        lstm_out, _ = self.lstm(x)
        context = self.attn(lstm_out)
        return self.fc(context)

# === Load model ===
model = CNN_BiLSTM_Attn()
model.load_state_dict(torch.load("emotion_model.pth", map_location=torch.device('cpu')))
model.eval()
print("✅ Model loaded. Sample FC weights:", list(model.fc.parameters())[0][0][:5])

# === Predict 10 samples ===
print("\n🔍 Running 10 Random Predictions:")
predicted = []

for i in range(10):
    idx = np.random.randint(0, len(X_raw))
    sample = normalize(X_raw[idx].reshape(1, -1)).astype(np.float32)
    sample_tensor = torch.tensor(sample).unsqueeze(2)

    with torch.no_grad():
        logits = model(sample_tensor)
        pred = torch.argmax(logits, dim=1).item()
        predicted.append(pred)

    print(f"[{i+1}] GT: {labels[y[idx]]:<8} | Pred: {labels[pred]:<8} | Logits: {logits.numpy().flatten()}")

print("\n📊 Prediction Count:", Counter(predicted))
