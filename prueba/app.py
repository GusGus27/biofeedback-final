import threading
import time
import json
import serial
import asyncio
import websockets
from collections import deque
from fastapi import FastAPI
from pyshimmer import ShimmerBluetooth, DEFAULT_BAUDRATE, DataPacket, EChannelType
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request

# Configuración fija
SHIMMER_PORT = "COM4"  # Cambia esto por el puerto real de tu Shimmer
SHIMMER_CHANNELS = ["INTERNAL_ADC_13", "GSR_RAW", "TIMESTAMP"]
WS_API_URL = "ws://localhost:8000/ws"  # URL del servicio de procesamiento

SAMPLING_RATE = 1024  # Hz

app = FastAPI()
templates = Jinja2Templates(directory="templates")
shimmers = {}
ws_loop = None
ws_queue = None

def start_ws_loop():
    global ws_loop, ws_queue
    ws_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(ws_loop)      # <-- PRIMERO el event loop
    ws_queue = asyncio.Queue(loop=ws_loop)           # <-- LUEGO la queue
    ws_loop.run_until_complete(ws_sender())

async def ws_sender():
    global ws_queue
    while True:
        try:
            async with websockets.connect(WS_API_URL) as websocket:
                while True:
                    data = await ws_queue.get()
                    await websocket.send(json.dumps(data))
        except Exception as e:
            print(f"Error en conexión WS: {e}")
            await asyncio.sleep(2)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/connect_shimmer")
async def connect_shimmer():
    global ws_loop, ws_queue
    device_name = "Shimmer1"
    port = SHIMMER_PORT
    channels = SHIMMER_CHANNELS

    if device_name in shimmers:
        return {"status": "error", "message": "Dispositivo ya conectado"}

    try:
        ser = serial.Serial(port, DEFAULT_BAUDRATE, timeout=2)
        shimmer = ShimmerBluetooth(ser)
        shimmer.initialize()
        shimmer.set_sampling_rate(SAMPLING_RATE)
        selected_channels = [EChannelType[name] for name in channels]

        def handle_pkt(pkt: DataPacket):
            PPG_CHANNEL = EChannelType.INTERNAL_ADC_13
            if PPG_CHANNEL in selected_channels:
                adc_value = pkt._values.get(PPG_CHANNEL)
                if adc_value is not None:
                    data = {
                        "device": device_name,
                        "timestamp": time.time(),
                        "adc_value": adc_value
                    }
                    # Enviar al ws_queue para que lo procese el hilo de ws
                    if ws_loop and ws_loop.is_running() and ws_queue:
                        asyncio.run_coroutine_threadsafe(ws_queue.put(data), ws_loop)

        shimmer.add_stream_callback(handle_pkt)
        shimmer.start_streaming()

        shimmers[device_name] = {
            "instance": shimmer,
            "channels": selected_channels
        }

        # Inicia el hilo de WebSocket si no está corriendo
        if not ws_loop or not ws_loop.is_running():
            threading.Thread(target=start_ws_loop, daemon=True).start()
            # Espera a que ws_queue esté lista
            while ws_queue is None:
                time.sleep(0.05)

        return {"status": "ok", "message": f"Conectado {device_name} en {port}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/disconnect_shimmer")
async def disconnect_shimmer():
    device_name = "Shimmer1"
    if device_name in shimmers:
        shimmer = shimmers[device_name]["instance"]
        shimmer.stop_streaming()
        shimmer.shutdown()
        del shimmers[device_name]
        return {"status": "ok", "message": "Desconectado"}
    else:
        return {"status": "error", "message": "Dispositivo no encontrado"}