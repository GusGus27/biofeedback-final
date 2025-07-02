import json
import numpy as np
from collections import deque
from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
from scipy.signal import find_peaks, butter, filtfilt
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # O restringe a ["http://localhost:8001"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Parámetros
SAMPLING_RATE = 1024
BUFFER_SIZE = SAMPLING_RATE * 30  # 30 segundos de datos

# Buffers y resultados
ppg_data = deque(maxlen=BUFFER_SIZE)
last_metrics = {"BPM": None, "IBI": None, "RMSSD": None}

def bandpass_filter(signal, fs, lowcut=0.5, highcut=5.0, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def compute_bpm(peaks, fs, duration_sec):
    if len(peaks) < 2:
        return None
    num_beats = len(peaks) - 1
    bpm = (num_beats / duration_sec) * 60
    return round(bpm, 1)

def compute_ibi_rmssd(peaks, fs):
    if len(peaks) < 2:
        return None, None
    ibi = np.diff(peaks) / fs * 1000  # en ms
    if len(ibi) < 2:
        return np.mean(ibi), None
    rmssd = np.sqrt(np.mean(np.diff(ibi) ** 2))
    return np.mean(ibi), round(rmssd, 2)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            data = await websocket.receive_text()
            pkt = json.loads(data)
            adc_value = pkt.get("adc_value")
            if adc_value is not None:
                ppg_data.append(adc_value)
                # Procesa cada vez que tengas suficiente data
                if len(ppg_data) == BUFFER_SIZE:
                    signal = np.array(ppg_data)
                    signal = signal - np.mean(signal)
                    filtered_signal = bandpass_filter(signal, SAMPLING_RATE)
                    peaks, _ = find_peaks(filtered_signal, distance=SAMPLING_RATE//2.5)
                    duration_sec = BUFFER_SIZE / SAMPLING_RATE
                    bpm = compute_bpm(peaks, SAMPLING_RATE, duration_sec)
                    ibi, rmssd = compute_ibi_rmssd(peaks, SAMPLING_RATE)
                    last_metrics["BPM"] = bpm
                    last_metrics["IBI"] = round(ibi, 1) if ibi is not None else None
                    last_metrics["RMSSD"] = rmssd
        except Exception as e:
            print(f"Error en WebSocket: {e}")
            break

@app.get("/last_metrics")
async def get_last_metrics():
    return JSONResponse(content=last_metrics)