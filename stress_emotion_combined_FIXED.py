# === Combined EEG Emotion + Stress Subtype Classifier (Fixed) ===
import numpy as np
import torch
import torch.nn as nn
import random
from sklearn.metrics import classification_report

# === Load both datasets ===
main_data = np.load("eeg_emotion_balanced_dataset.npz")
sub_data = np.load("eeg_stress_subtype_balanced.npz")

X_main = main_data["X"]
y_main = main_data["y"]  # 0=Relaxed, 1=Focused, 2=Stressed, 3=Drowsy
X_sub = sub_data["X"]
y_sub = sub_data["y"]    # 0=Acute, 1=Cognitive

# === Normalize ===
def normalize(X):
    return (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)

X_main = normalize(X_main)
X_sub = normalize(X_sub)

X_main = np.expand_dims(X_main.astype(np.float32), axis=2)
X_sub = np.expand_dims(X_sub.astype(np.float32), axis=2)

# === Attention Layer ===
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, 1)

    def forward(self, lstm_output):
        weights = torch.softmax(self.attn(lstm_output), dim=1)
        context = torch.sum(weights * lstm_output, dim=1)
        return context

# === Emotion Model ===
class BiLSTM_Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 32, bidirectional=True, batch_first=True)
        self.attn = Attention(32)
        self.fc = nn.Linear(64, 4)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context = self.attn(lstm_out)
        return self.fc(context)

# === Stress Subtype Classifier ===
class StressSubtypeClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 32, bidirectional=True, batch_first=True)
        self.attn = Attention(32)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context = self.attn(lstm_out)
        return self.fc(context)

# === Load Models ===
emotion_model = BiLSTM_Attn()
stress_model = StressSubtypeClassifier()

emotion_model.load_state_dict(torch.load("emotion_model_verified.pth", map_location=torch.device('cpu')))
stress_model.load_state_dict(torch.load("stress_subtype_model.pth", map_location=torch.device('cpu')))

emotion_model.eval()
stress_model.eval()

# === Prediction Function ===
def classify_emotion_with_subtype(eeg_input):
    eeg_input = normalize(eeg_input.reshape(1, -1))
    eeg_input = torch.tensor(eeg_input.astype(np.float32)).unsqueeze(2)

    with torch.no_grad():
        emotion_logits = emotion_model(eeg_input)
        emotion_class = torch.argmax(emotion_logits, dim=1).item()

        if emotion_class == 2:
            stress_logits = stress_model(eeg_input)
            stress_class = torch.argmax(stress_logits, dim=1).item()
            stress_label = "Acute Stress" if stress_class == 0 else "Cognitive Stress"
            return f"Stressed ({stress_label})"
        else:
            return ["Relaxed", "Focused", "Stressed", "Drowsy"][emotion_class]

# === Run Example ===
idx = random.randint(0, len(X_main) - 1)
example = main_data["X"][idx]  # raw unnormalized input
labels = ["Relaxed", "Focused", "Stressed", "Drowsy"]
print(f"Ground Truth Label: {y_main[idx]} → {labels[y_main[idx]]}")
result = classify_emotion_with_subtype(example)
print("Predicted State:", result)
