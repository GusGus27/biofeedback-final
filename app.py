from StressDetector import StressDetector
from collections import deque
import time
import threading
import serial
import asyncio
import json
from pyshimmer import ShimmerBluetooth, DEFAULT_BAUDRATE, DataPacket, EChannelType
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request, APIRouter
from serial.tools import list_ports
import uvicorn
from collections import deque
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
from fastapi.responses import HTMLResponse
from typing import Optional

# Parámetros
SAMPLING_RATE = 1024  # Hz
BUFFER_SIZE = SAMPLING_RATE * 10  # 5 segundos de datos
UPDATE_INTERVAL = 2  # segundos

ppg_data = {}
bpm_values = {}
ibi_values = {}
ibi_series = {}
peak_timestamps = {}
rmssd_history = {}
rmssd_values = {}
bpm_history = {}

############# JULER #############
# ---------- EDA / Stress ----------
EDA_FREQ = 1024              # muestras/s que declaras en la UI
WIN_SEC  = 60                # ventana 60 s
STRIDE_S = 30                # refresco cada 30 s
WIN_SAMP = EDA_FREQ * WIN_SEC * 6
STRD_SAMP = EDA_FREQ * STRIDE_S

eda_buffers   = {}           # device -> deque(maxlen=WIN_SAMP)
stress_levels = {}
latest_payload = {}           # device -> último nivel 1-5
stride_count  = {}           # device -> muestras acumuladas
detectors     = {}           # device -> StressDetector
# ----------------------------------
############# JULER #############

channel_aliases = {
    "INTERNAL_ADC_13": "PPG"
}

def gsr_raw_to_conductance(raw_value):
    range_setting = (raw_value >> 14) & 0b11
    adc_value = raw_value & 0x0FFF

    rf_values = {
        0: 40200,
        1: 287000,
        2: 1000000,
        3: 3300000
    }

    Rf = rf_values.get(range_setting, None)
    if Rf is None:
        return None

    V_adc = (adc_value / 4095.0) * 3.0
    if V_adc <= 0.5:
        return None

    Rs = (Rf * (3.0 - V_adc)) / (V_adc - 0.5)
    if Rs == 0:
        return None

    G_microS = 1e6 / Rs
    return G_microS


app = FastAPI()
templates = Jinja2Templates(directory="templates")

shimmers = {}

event_loop = None

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                self.disconnect(connection)

manager = ConnectionManager()

# Función de filtro bandpass para PPG (0.5-5 Hz)
def bandpass_filter(signal, fs, lowcut=0.5, highcut=5):
    b, a = butter(2, [lowcut / (0.5 * fs), highcut / (0.5 * fs)], btype='band')
    return filtfilt(b, a, signal)

def compute_ibi_rmssd(ppg_signal, fs):
    ppg_signal = np.array(ppg_signal).flatten()
    if len(ppg_signal) < 2:
        return np.nan, np.nan
    # Detección de picos
    peak_indices, _ = find_peaks(ppg_signal, distance=fs*0.4, prominence=(np.std(ppg_signal) * 0.5, None))
    if len(peak_indices) < 2:
        return np.nan, np.nan
    ibi = np.diff(peak_indices) / fs  # en segundos
    ibi_ms = ibi * 1000  # en milisegundos
    rmssd = np.sqrt(np.mean(np.diff(ibi_ms) ** 2)) if len(ibi_ms) > 1 else np.nan
    ibi_mean = np.mean(ibi_ms)
    return ibi_mean, rmssd

@app.on_event("startup")
async def startup_event():
    global event_loop
    event_loop = asyncio.get_running_loop()

@app.get("/live", response_class=HTMLResponse)
async def live_visualization():
    with open("templates/live.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/")
def get_html(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/available_ports")
async def available_ports():
    ports = list_ports.comports()
    return [{"port": p.device, "name": p.description} for p in ports]

@app.post("/connect_shimmer")
async def connect_shimmer(request: Request):
    form = await request.form()
    device_name = form['device_name']
    port = form['port']
    channels = form['channels'].split(',')

    if device_name in shimmers:
        return {"status": "error", "message": "Dispositivo ya conectado"}

    ser = serial.Serial(port, DEFAULT_BAUDRATE, timeout=2)
    shimmer = ShimmerBluetooth(ser)
    shimmer.initialize()
    shimmer.set_sampling_rate(SAMPLING_RATE)

    selected_channels = [EChannelType[name.strip()] for name in channels]


############# JULER #############
    EDA_CHANNEL = EChannelType.GSR_RAW  # usa el canal GSR real de tu Shimmer
    if EDA_CHANNEL in selected_channels:
        eda_buffers[device_name] = deque(maxlen=WIN_SAMP)
        stride_count[device_name] = 0
        stress_levels[device_name] = 0           # neutral inicial
        detectors[device_name] = StressDetector(
            _interval=3,                 # tercio de la ventana
            _samples_per_minute=EDA_FREQ*60
        )
############# JULER #############


    if EChannelType.INTERNAL_ADC_13 in selected_channels:
        ppg_data[device_name] = deque(maxlen=BUFFER_SIZE)
        bpm_values[device_name] = 0
        bpm_history[device_name] = deque(maxlen=5)
        peak_timestamps[device_name] = deque(maxlen=300)
        rmssd_history[device_name] = deque(maxlen=5)
        ibi_series[device_name] = deque(maxlen=20)
        shimmer_state = {"last_bpm_time": time.time()}

    def handle_pkt(pkt: DataPacket):
        values = {
            device_name: {
                ch.name: pkt._values.get(ch, "NA") for ch in selected_channels
            }
        }

        latest_payload[device_name] = values[device_name]

        if EChannelType.INTERNAL_ADC_13 in selected_channels:
            adc_value = pkt._values.get(EChannelType.INTERNAL_ADC_13)
            
            if adc_value is not None:
                ppg_data[device_name].append(adc_value)

                # Cálculo BPM cada UPDATE_INTERVAL segundos
                now = time.time()
                if len(ppg_data[device_name]) == BUFFER_SIZE and (now - shimmer_state["last_bpm_time"] >= UPDATE_INTERVAL):
                    signal = np.array(ppg_data[device_name])
                    signal = signal - np.mean(signal)
                    filtered_signal = bandpass_filter(signal, SAMPLING_RATE)

                    # BPM
                    peaks, _ = find_peaks(filtered_signal, distance=SAMPLING_RATE * 0.4)
                    bpm = (len(peaks) / (BUFFER_SIZE / SAMPLING_RATE)) * 60
                    bpm_history[device_name].append(bpm)
                    smoothed_bpm = np.mean(bpm_history[device_name])
                    bpm_values[device_name] = round(smoothed_bpm, 1)

                    # --- IBI: cálculo rápido instantáneo (como antes) ---
                    ibi, _ = compute_ibi_rmssd(filtered_signal, SAMPLING_RATE)
                    ibi_values[device_name] = round(ibi, 1) if not np.isnan(ibi) else None

                    # --- HRV (RMSSD): basado en timestamp de picos acumulados ---
                    # Detección de picos actual
                    peaks, _ = find_peaks(filtered_signal, distance=SAMPLING_RATE * 0.4)
                    peak_times = peaks / SAMPLING_RATE  # tiempo en segundos relativo al inicio del buffer

                    # Calcular IBIs y añadir a ibi_series
                    if len(peak_times) >= 2:
                        ibis = np.diff(peak_times) * 1000  # en ms
                        for ibi in ibis:
                            ibi_series[device_name].append(ibi)

                    # Calcular RMSSD si hay suficientes IBIs en ibi_series
                    if len(ibi_series[device_name]) >= 6:
                        rmssd = np.sqrt(np.mean(np.diff(np.array(ibi_series[device_name])) ** 2))
                        rmssd_history[device_name].append(rmssd)
                        smoothed_rmssd = np.mean(rmssd_history[device_name])
                        rmssd_values[device_name] = round(smoothed_rmssd, 1)
                    else:
                        rmssd_values[device_name] = None

                    shimmer_state["last_bpm_time"] = now
        ########### JULER #############
        # ---------- EDA streaming ----------
        EDA_CHANNEL = EChannelType.GSR_RAW
        if EDA_CHANNEL in selected_channels:
            gsr_raw_value = pkt._values.get(EDA_CHANNEL)
            if gsr_raw_value is not None:

                conductance_microS = gsr_raw_to_conductance(gsr_raw_value)

                if conductance_microS is not None:
                    eda_buffers[device_name].append(conductance_microS)
                    stride_count[device_name] += 1
                
                values[device_name]["Skin_Conductance_uS"] = round(conductance_microS, 2) if conductance_microS else None

                if len(eda_buffers[device_name]) == WIN_SAMP and stride_count[device_name] >= STRD_SAMP:
                    print(f"\nProcesando GSR para {device_name} con {len(eda_buffers[device_name])} muestras\n")
                    print(f"\nLista completa de GSR: {eda_buffers[device_name]}\n")
                    detectors[device_name].process_gsr(list(eda_buffers[device_name]))
                    print("Lista de buffers:", list(eda_buffers[device_name]))
                    sax_list = detectors[device_name].get_sax_values()
                    print(f"\nSAX List: {sax_list}\n")
                    stress_levels[device_name] = sax_list[-1] if sax_list else 0
                    stride_count[device_name] = 0
                #else:
                    #print(f"Esperando {WIN_SAMP} muestras de GSR para {device_name}, muestras actuales: {len(eda_buffers[device_name])}")
            else:
                print(f"Canal {EDA_CHANNEL.name} no encontrado en el paquete de datos.")
        # ------------------------------------
        ########### JULER #############

        values[device_name]["BPM"] = bpm_values.get(device_name)
        values[device_name]["IBI"] = ibi_values.get(device_name)
        values[device_name]["RMSSD"] = rmssd_values.get(device_name)
        ########### JULER #############
        values[device_name]["Stress"] = stress_levels.get(device_name)
        ########### JULER #############

        message = json.dumps(values)
        asyncio.run_coroutine_threadsafe(manager.broadcast(message), event_loop)

    shimmer.add_stream_callback(handle_pkt)
    shimmer.start_streaming()

    shimmers[device_name] = {
        "instance": shimmer,
        "channels": selected_channels
    }

    threading.Thread(target=lambda: time.sleep(1)).start()

    return {"status": "ok", "message": f"Conectado {device_name} en {port}"}

@app.post("/disconnect_shimmer")
async def disconnect_shimmer(request: Request):
    form = await request.form()
    device_name = form['device_name']

    if device_name in shimmers:
        shimmer = shimmers[device_name]["instance"]
        shimmer.stop_streaming()
        shimmer.shutdown()
        del shimmers[device_name]
        return {"status": "ok", "message": f"{device_name} desconectado correctamente"}
    else:
        return {"status": "error", "message": f"{device_name} no está conectado"}

@app.get("/detect_channels")
def detect_channels(port: str):
    try:
        ser = serial.Serial(port, DEFAULT_BAUDRATE, timeout=2)
        shimmer = ShimmerBluetooth(ser)
        shimmer.initialize()

        detected = set()

        def callback(pkt: DataPacket):
            detected.update(pkt._values.keys())

        shimmer.add_stream_callback(callback)
        shimmer.start_streaming()
        time.sleep(4)
        shimmer.stop_streaming()
        shimmer.shutdown()
        ser.close()

        return {"channels": [ch.name for ch in detected]}

    except Exception as e:
        return {"error": str(e)}
    
@app.get("/last_metrics")
def last_metrics(device: Optional[str] = None):
    if device:
        return latest_payload.get(device, {})
    return latest_payload

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # Puedes ignorar lo recibido
    except:
        print("Cliente desconectado")
    finally:
        manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

