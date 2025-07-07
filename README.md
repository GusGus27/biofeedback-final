# Biofeedback Final

Este proyecto permite la visualización en tiempo real de señales fisiológicas (como PPG y GSR) obtenidas desde un dispositivo Shimmer, con procesamiento y detección de estrés.

## Requisitos

- Python 3.8
- Tener emparejado previamente un dispositivo **Shimmer** vía Bluetooth con la computadora o laptop desde donde se va a conectar.
- El Shimmer debe estar configurado para transmitir (stream) los canales que se desean visualizar en la página.

## Instalación

1. Clona este repositorio.
2. Instala las dependencias ejecutando:

```bash
pip install -r requirements.txt
```

## Uso

1. Ejecuta la aplicación principal:

```bash
python app.py
```

2. Abre tu navegador y accede a `http://localhost:8000`.
3. En la página desplegada:
   - Selecciona el puerto correspondiente al Shimmer.
   - Selecciona los canales que deseas visualizar.
   - Haz clic en conectar para comenzar la transmisión y visualización de datos.

## Notas importantes

- **Emparejamiento:** Antes de usar la aplicación, asegúrate de que el Shimmer esté emparejado vía Bluetooth con tu computadora.
- **Configuración del Shimmer:** El dispositivo debe estar configurado para transmitir los canales que quieras visualizar.
- **Ventanas de cálculo:**
  - Los cálculos de **HR** (frecuencia cardíaca), **HRV** (variabilidad de la frecuencia cardíaca, RMSSD) e **IBI** (intervalo entre latidos) no se mostrarán hasta pasados **30 segundos** desde el inicio de la transmisión, ya que las ventanas de análisis son de esa duración. Estos valores se recalculan cada 2 segundos.
  - El cálculo del **estrés** requiere una ventana de **6 minutos** de datos y se actualiza cada 30 segundos.

## Endpoints principales

- `GET /` : Página principal con la interfaz de usuario.
- `GET /live` : Visualización en vivo de las señales.
- `GET /available_ports` : Devuelve los puertos seriales disponibles para conectar el Shimmer.
- `POST /connect_shimmer` : Permite conectar un dispositivo Shimmer especificando el puerto y los canales.
- `POST /disconnect_shimmer` : Desconecta el dispositivo Shimmer.
- `GET /detect_channels` : Detecta los canales disponibles en un puerto dado.
- `GET /last_metrics` : Devuelve las últimas métricas recibidas para un dispositivo.
- `WS /ws` : WebSocket para transmisión en tiempo real de los datos.

## Breve explicación de StressDetector

El módulo `StressDetector` procesa la señal de conductancia de la piel (GSR) para detectar niveles de estrés. Utiliza técnicas de filtrado, normalización y discretización (SAX) sobre ventanas de 6 minutos, actualizando el nivel de estrés cada 30 segundos.
