import numpy as np
from scipy.signal import welch
from pythonosc import dispatcher, osc_server
from collections import deque
import time

# === Settings ===
SAMPLE_RATE = 256
BUFFER_DURATION = 2
CHANNEL_INDEX = 1  # AF7
PORT = 5000
DATA_SAVE_PATH = "eeg_features_dataset.npz"

eeg_buffer = deque(maxlen=SAMPLE_RATE * BUFFER_DURATION)
collected_X = []
collected_y = []

def get_band_features(signal, fs=SAMPLE_RATE):
    freqs, psd = welch(signal, fs, nperseg=fs)
    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 12),
        "beta": (12, 30),
        "gamma": (30, 45)
    }
    powers = {}
    for band, (low, high) in bands.items():
        idx = np.logical_and(freqs >= low, freqs <= high)
        powers[band] = np.mean(psd[idx])

    # Ratios
    alpha = powers['alpha'] + 1e-6
    theta = powers['theta']
    beta = powers['beta']

    features = [
        powers['delta'],
        powers['theta'],
        alpha,
        beta,
        powers['gamma'],
        beta / alpha,
        theta / alpha,
        beta / (alpha + theta + 1e-6)
    ]

    return np.array(features), beta / alpha

def label_stress(score):
    if score < 1.0:
        return 0  # Positive
    elif score < 2.0:
        return 1  # Acute
    elif score < 3.5:
        return 2  # Episodic
    else:
        return 3  # Toxic

def process_eeg(unused_addr, *args):
    try:
        val = float(args[CHANNEL_INDEX])
        eeg_buffer.append(val)
    except:
        pass

def collector_loop():
    last_time = time.time()
    while True:
        if len(eeg_buffer) == eeg_buffer.maxlen:
            signal = np.array(eeg_buffer)
            features, stress_score = get_band_features(signal)
            label = label_stress(stress_score)

            collected_X.append(features)
            collected_y.append(label)

            print(f"[{time.strftime('%H:%M:%S')}] Saved | Stress Score: {stress_score:.2f} → Label: {label}")
            eeg_buffer.clear()

        # Save every 10s
        if time.time() - last_time > 10:
            if len(collected_X) > 0:
                np.savez(DATA_SAVE_PATH, X=np.array(collected_X), y=np.array(collected_y))
                print(f"💾 Saved {len(collected_X)} samples to {DATA_SAVE_PATH}")
                last_time = time.time()

if __name__ == "__main__":
    print("📥 Starting EEG Feature Collection with Auto Labeling")
    dispatcher = dispatcher.Dispatcher()
    dispatcher.map("/muse/eeg", process_eeg)

    server = osc_server.ThreadingOSCUDPServer(("0.0.0.0", PORT), dispatcher)
    from threading import Thread
    Thread(target=collector_loop, daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Exited.")
