import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from collections import deque
from pythonosc import dispatcher, osc_server
from threading import Thread
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# === Emotion Mapping & Logging ===
emotion_map = {'Drowsy': 0, 'Relaxed': 0.33, 'Focused': 0.66, 'Stressed': 1}
emotion_log = []

def log_emotion(emotion, confidence=None):
    timestamp = datetime.now().strftime("%H:%M:%S")
    if confidence:
        print(f"[{timestamp}] → ML: {emotion} ({confidence:.2f})")
    else:
        print(f"[{timestamp}] → Rule: {emotion}")
    emotion_log.append((timestamp, emotion))

# === Buffers & Calibration ===
WINDOW_SIZE = 5
theta_buf, alpha_buf, beta_buf, gamma_buf, delta_buf = [deque(maxlen=WINDOW_SIZE) for _ in range(5)]

# Replace these with training data's mean & std
mean = np.array([800, 800, 800, 800, 600], dtype=np.float32)
std = np.array([50, 50, 50, 50, 150], dtype=np.float32)

# === Threshold-based fallback classifier ===
def normalize_bands(theta, alpha, beta, gamma, delta):
    total = theta + alpha + beta + gamma + delta
    return {
        'theta': theta / total,
        'alpha': alpha / total,
        'beta': beta / total,
        'gamma': gamma / total,
        'delta': delta / total
    }


def formula_based_classify(theta, alpha, beta, gamma, delta):
    epsilon = 1e-6
    drowsy_index = theta / (beta + epsilon)
    relaxed_index = alpha / (beta + epsilon)
    focus_index = beta / (alpha + theta + epsilon)
    stress_index = (beta + gamma) / (alpha + epsilon)

    if drowsy_index > 4 and theta > alpha:
        return "Drowsy"
    elif relaxed_index > 2 and alpha > theta and alpha > beta:
        return "Relaxed"
    elif focus_index > 1 and beta > alpha and beta > theta:
        return "Focused"
    elif stress_index > 2 and gamma > alpha:
        return "Stressed"
    else:
        return "Uncertain"


def rule_based_classify(theta, alpha, beta, gamma, delta):
    bands = normalize_bands(theta, alpha, beta, gamma, delta)
    θ, α, β, γ = bands['theta'], bands['alpha'], bands['beta'], bands['gamma']
    if θ > 0.28 and β < 0.10: return "Drowsy"
    elif α > 0.30 and β < 0.15: return "Relaxed"
    elif β > 0.20 and α < 0.20 and θ < 0.20: return "Focused"
    elif β > 0.20 and γ > 0.30 and α < 0.15: return "Stressed"
    else: return "Uncertain"

def get_smoothed(buf): return np.mean(buf) if buf else 0

# === Model Definition ===
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, 1)
    def forward(self, lstm_out):
        weights = torch.softmax(self.attn(lstm_out), dim=1)
        return torch.sum(weights * lstm_out, dim=1)

class CNN_BiLSTM_Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, bidirectional=True, batch_first=True)
        self.attn = Attention(64)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, 4)
        nn.init.xavier_uniform_(self.fc.weight)
    def forward(self, x):  # x: (B, 1, 5)
        x = self.cnn(x)        # (B, 32, 5)
        x = x.permute(0, 2, 1) # (B, 5, 32)
        out, _ = self.lstm(x)  # (B, 5, 128)
        x = self.attn(out)
        x = self.dropout(x)
        return self.fc(x)

# === Load Trained Model ===
device = torch.device("cpu")
model = CNN_BiLSTM_Attn().to(device)
model.load_state_dict(torch.load("emotion_model_new.pth", map_location=device))
model.eval()
print("[INFO] Model loaded.")

# === EEG Data Handler ===
def process_eeg(address, *args):
    if len(args) != 5:
        print(f"[ERROR] Expected 5 values, got {len(args)}")
        return
    theta, alpha, beta, gamma, delta = args
    theta_buf.append(theta)
    alpha_buf.append(alpha)
    beta_buf.append(beta)
    gamma_buf.append(gamma)
    delta_buf.append(delta)

# === Real-time Prediction Thread ===
def realtime_classifier():
    while True:
        if len(theta_buf) >= WINDOW_SIZE:
            θ = get_smoothed(theta_buf)
            α = get_smoothed(alpha_buf)
            β = get_smoothed(beta_buf)
            γ = get_smoothed(gamma_buf)
            δ = get_smoothed(delta_buf)

            # Normalize input for model
            eeg = np.array([θ, α, β, γ, δ], dtype=np.float32)
            eeg = (eeg - mean) / (std + 1e-6)
            eeg = torch.tensor(eeg).view(1, 1, 5).to(device)

            try:
                with torch.no_grad():
                    output = model(eeg)
                    probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                    label = ["Relaxed", "Focused", "Stressed", "Drowsy"][np.argmax(probs)]
                    log_emotion(label, probs[np.argmax(probs)])
            except Exception as e:
                fallback = formula_based_classify(θ, α, β, γ, δ)
                log_emotion(fallback)

            time.sleep(1)

# === Live Plotting (Optional) ===
fig, ax = plt.subplots()
xdata, ydata = [], []
line, = ax.plot([], [], lw=2)
ax.set_ylim(-0.1, 1.1)
ax.set_xlim(0, 20)
ax.set_yticks([0, 0.33, 0.66, 1])
ax.set_yticklabels(['Drowsy', 'Relaxed', 'Focused', 'Stressed'])
ax.set_title("Real-Time Emotion")
ax.set_ylabel("Class")
ax.grid()

def update_plot(frame):
    if emotion_log:
        _, latest_emotion = emotion_log[-1]
        xdata.append(len(xdata))
        ydata.append(emotion_map.get(latest_emotion, 0))
        line.set_data(xdata[-20:], ydata[-20:])
        ax.set_xlim(max(0, len(xdata) - 20), len(xdata))
    return line,



# === Start Server ===
if __name__ == "__main__":
    from pythonosc import dispatcher as disp
    from pythonosc import osc_server
    import time

    PORT = 5000
    IP = "0.0.0.0"

    print("📡 Listening to EEG stream on /muse/eeg")
    dispatcher = disp.Dispatcher()
    dispatcher.map("/muse/eeg", process_eeg)
    server = osc_server.ThreadingOSCUDPServer((IP, PORT), dispatcher)
    Thread(target=realtime_classifier, daemon=True).start()

    try:
        print("🧠 Starting Real-Time Emotion Prediction")
        plt.show()
        server.serve_forever()
    except KeyboardInterrupt:
        print("🛑 Exiting Real-Time Predictor") 
