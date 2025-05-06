# 🧠 Muse EEG Real-Time Emotion Detection

Real-time EEG-based Emotion & Stress Classification using Muse headband data via OSC (Open Sound Control).  
This project uses PyTorch, SciPy, and real-time signal processing to classify mental states such as **Relaxed**, **Focused**, **Stressed**, and **Drowsy**.

A real-time brainwave emotion classification system built with:

- **Muse 2 + Mind Monitor** for EEG signal streaming
- **Python (OSC + WebSocket)** for signal processing + emotion classification
- **Next.js (Vercel)** as the interactive frontend dashboard
- **Render** for backend WebSocket server hosting

## 🚀 Live Frontend

🌐 [https://v0-new-project-wum9kvc3eqx.vercel.app](https://v0-new-project-wum9kvc3eqx.vercel.app)

## 📡 WebSocket Endpoint

🧩 `wss://muse-eeg-backend.onrender.com` (automatically receives and broadcasts real-time classified emotional state)

---

## 📦 Project Structure

