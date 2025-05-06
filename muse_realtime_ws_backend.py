
import asyncio
import websockets
from datetime import datetime
from pythonosc import dispatcher
from pythonosc import osc_server
import threading
import os
# Global variable to hold latest emotion
latest_emotion = "Unknown"

# Formula-based emotion classification
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

# WebSocket broadcasting
connected_clients = set()

async def broadcast_emotion():
    while True:
        if connected_clients:
            for ws in list(connected_clients):
                try:
                    await ws.send(latest_emotion)
                except websockets.ConnectionClosed:
                    connected_clients.remove(ws)
        await asyncio.sleep(1)

async def ws_handler(websocket, path):
    connected_clients.add(websocket)
    print("WebSocket client connected.")
    try:
        await websocket.wait_closed()
    finally:
        connected_clients.remove(websocket)
        print("WebSocket client disconnected.")

# OSC EEG handler
def eeg_handler(address: str, *args):
    global latest_emotion
    if len(args) >= 5:
        theta, alpha, beta, gamma, delta = args[:5]
        emotion = formula_based_classify(theta, alpha, beta, gamma, delta)
        latest_emotion = emotion
        print(f"{datetime.now()} - EMOTION: {emotion}")

# Start OSC server in background thread
def start_osc_server(ip="0.0.0.0", port=5000):
    disp = dispatcher.Dispatcher()
    disp.map("/muse/eeg", eeg_handler)
    server = osc_server.ThreadingOSCUDPServer((ip, port), disp)
    print(f"OSC server listening on {ip}:{port}")
    server.serve_forever()

# Run both OSC + WebSocket server
def main():
    port = int(os.environ.get("PORT", 10000))  # Render injects the PORT variable

    threading.Thread(target=start_osc_server, daemon=True).start()
    print(f"WebSocket server starting on ws://0.0.0.0:{port}")
    
    loop = asyncio.get_event_loop()
    loop.run_until_complete(websockets.serve(ws_handler, "0.0.0.0", port))
    loop.run_until_complete(broadcast_emotion())
    loop.run_forever()

if __name__ == "__main__":
    main()