import numpy as np
from scipy.signal import welch
from pythonosc import dispatcher, osc_server
import torch
from collections import deque
from threading import Thread
import time

# === Settings ===
SAMPLE_RATE = 256
BUFFER_DURATION = 2  # seconds
CHANNEL_INDEX = 1
PORT = 5000
LABELS = ["Relaxed", "Focused", "Stressed", "Drowsy"]
MODEL_PATH = "new_try.pth"

# === EEG Buffer ===
eeg_buffer = deque(maxlen=SAMPLE_RATE * BUFFER_DURATION)

# === Preprocessing ===
def normalize(X):
    return (X - np.mean(X)) / (np.std(X) + 1e-6)

# === Model Architecture ===
class Attention(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = torch.nn.Linear(hidden_size * 2, 1)
    def forward(self, lstm_output):
        weights = torch.softmax(self.attn(lstm_output), dim=1)
        return torch.sum(weights * lstm_output, dim=1)

class CNN_BiLSTM_Attn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv1d(1, 16, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU()
        )
        self.lstm = torch.nn.LSTM(16, 32, bidirectional=True, batch_first=True)
        self.attn = Attention(32)
        self.dropout = torch.nn.Dropout(0.3)
        self.fc = torch.nn.Linear(64, 4)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        context = self.attn(lstm_out)
        context = self.dropout(context)
        return self.fc(context)

# === Load Model ===
model = CNN_BiLSTM_Attn()
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()
print("✅ Model loaded.")

# === Optional Band Power-Based Thresholding Logic ===
def classify_by_band_power(theta, alpha, beta, gamma):
    if theta > 6 and alpha <= 6 and beta <= 5 and gamma <= 10:
        return "Drowsy"
    elif 4 <= theta <= 6 and 9 <= alpha <= 12 and 6 <= beta <= 10 and gamma <= 15:
        return "Relaxed"
    elif theta <= 4 and alpha <= 6 and 15 <= beta <= 25 and 15 <= gamma <= 30:
        return "Focused"
    elif theta <= 4 and alpha <= 5 and 20 <= beta <= 30 and 30 <= gamma <= 50:
        return "Stressed"
    else:
        return "Unknown"

def compute_bandpowers(eeg_segment):
    f, Pxx = welch(eeg_segment, fs=SAMPLE_RATE, nperseg=256)
    def band_power(f_low, f_high):
        mask = (f >= f_low) & (f <= f_high)
        return np.mean(Pxx[mask]) if np.any(mask) else 0
    return (
        band_power(4, 7),    # Theta
        band_power(8, 13),   # Alpha
        band_power(13, 30),  # Beta
        band_power(30, 100)  # Gamma
    )

# === EEG Data Callback ===
def process_eeg(unused_addr, *args):
    try:
        val = float(args[CHANNEL_INDEX])
        eeg_buffer.append(val)
    except Exception:
        pass

# === Real-Time Prediction Loop ===
def realtime_classifier():
    while True:
        if len(eeg_buffer) == eeg_buffer.maxlen:
            raw_eeg = np.array(eeg_buffer).astype(np.float32)

            # Model prediction
            norm = normalize(raw_eeg).reshape(1, 1, -1)
            with torch.no_grad():
                logits = model(torch.tensor(norm))
                probs = torch.softmax(logits, dim=1).squeeze().numpy()
                pred_class = np.argmax(probs)
                confidence = probs[pred_class]

            # EEG threshold classification
            theta, alpha, beta, gamma = compute_bandpowers(raw_eeg)
            band_class = classify_by_band_power(theta, alpha, beta, gamma)

            print(f"[{time.strftime('%H:%M:%S')}] → ML: {LABELS[pred_class]} ({confidence:.2f}) | Band: {band_class} | θ={theta:.2f}, α={alpha:.2f}, β={beta:.2f}, γ={gamma:.2f}")
            eeg_buffer.clear()

        time.sleep(20)

# === Start OSC Server ===
if __name__ == "__main__":
    print("📡 Listening to EEG stream on /muse/eeg")
    dispatcher = dispatcher.Dispatcher()
    dispatcher.map("/muse/eeg", process_eeg)
    server = osc_server.ThreadingOSCUDPServer(("0.0.0.0", PORT), dispatcher)
    Thread(target=realtime_classifier, daemon=True).start()

    try:
        print("🧠 Starting Real-Time Emotion Prediction")
        server.serve_forever()
    except KeyboardInterrupt:
        print("🛑 Exiting Real-Time Predictor")
