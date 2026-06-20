# Filtro Digital FIR en Python para Señales SDR
**Universidad Nacional de Chimborazo — Escuela de Telecomunicaciones**  
**Curso:** Electrónica II  
**Proyecto:** Diseño e implementación de filtros digitales FIR con ADALM-Pluto  

---

## Descripción

Este proyecto diseña e implementa filtros digitales FIR (Finite Impulse Response) en Python usando `scipy.signal`, aplicados a señales I/Q capturadas del ADALM-Pluto SDR. Se incluyen filtros paso bajo, paso alto y paso banda, con validación experimental mediante transmisión y recepción en 915 MHz.

---

## Estructura del repositorio

```
proyecto-filtros-fir/
├── python/            # Scripts Python por semana
├── figures/           # Figuras generadas en PDF
├── data/              # Datos experimentales (.csv)
├── docs/              # Informe LaTeX (Overleaf)
└── README.md
```

---

## Especificaciones técnicas

| Parámetro | Valor |
|---|---|
| Sample rate | 1 MHz |
| Frecuencia central TX/RX | 915 MHz |
| Dispositivo SDR | ADALM-Pluto |
| Señal de prueba | Multi-tono: 50 + 200 + 300 kHz |
| Ventana FIR | Hamming |
| Taps evaluados | 32, 64, 128 |

### Filtros diseñados

| Filtro | Frecuencia de corte | Taps | Atenuación SB |
|---|---|---|---|
| Paso bajo | 100 kHz | 65 | 59.0 dB |
| Paso alto | 200 kHz | 65 | 60.3 dB |
| Paso banda | 150–250 kHz | 65 | 93.2 dB |

---

## Requisitos

### Hardware
- ADALM-Pluto SDR
- Cable USB de datos (micro-USB)
- 2 antenas SMA

### Software
- Python 3.x
- Drivers PlutoSDR para Windows: https://wiki.analog.com/university/tools/pluto/drivers/windows

### Instalación de dependencias

```bash
pip install pyadi-iio numpy scipy matplotlib
```

---

## Cómo reproducir el proyecto

### 1. Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/proyecto-filtros-fir.git
cd proyecto-filtros-fir
```

### 2. Instalar dependencias
```bash
pip install pyadi-iio numpy scipy matplotlib
```

### 3. Conectar el ADALM-Pluto
- Conectar el Pluto al PC via USB
- Verificar conexión:
```bash
ping 192.168.2.1
```

### 4. Ejecutar los scripts en orden

| Semana | Script | Descripción |
|---|---|---|
| 1 | `python/semana1_Diseño de filtro paso bajo con scipy.signal.py` | Diseño filtro paso bajo |
| 2 | `python/semana2_Diseño de filtros paso alto y paso banda.py` | Filtros paso alto y paso banda |
| 3 | `python/semana3_Generación de señales multi-tono, transmisión con Pluto.py` | Transmisión multi-tono |
| 4 | `python/semana4_Captura de señales, aplicación de filtros, FFT.py` | Captura y filtrado con Pluto |
| 5 | `python/semana5_Comparación de órdenes de filtros (32, 64, 128 taps).py` | Comparación 32/64/128 taps |
| 6 | `python/semana6_Validación: verificación de atenuación y ganancia.py` | Validación experimental |
| 7 | `python/semana7_Análisis de resultados, gráficas comparativas.py` | Análisis y gráficas finales |

---

## Resultados principales

### Validación teórico vs experimental

| Filtro | Tono | Teórico (dB) | Experimental (dB) | Error (dB) |
|---|---|---|---|---|
| Paso bajo | 50 kHz | 0.0 | 0.0 | 0.0 |
| Paso bajo | 200 kHz | -92.3 | -84.2 | 8.1 |
| Paso bajo | 300 kHz | -59.5 | -59.4 | 0.1 |
| Paso alto | 50 kHz | -60.3 | -60.1 | 0.2 |
| Paso alto | 300 kHz | 0.0 | 0.0 | 0.0 |
| Paso banda | 200 kHz | 0.0 | 0.0 | 0.0 |

### Comparación de órdenes (filtro paso bajo)

| Taps | Atenuación SB | Ripple PB | Transición | Tiempo proc. |
|---|---|---|---|---|
| 32 | 59.0 dB | 1.84 dB | 59.0 kHz | 0.355 ms |
| 64 | 92.3 dB | 0.28 dB | 29.6 kHz | 0.284 ms |
| 128 | 67.1 dB | 0.03 dB | 14.8 kHz | 0.347 ms |

---

## Autores

- [Nombre del integrante 1] — Universidad Nacional de Chimborazo
- [Nombre del integrante 2] — Universidad Nacional de Chimborazo

---
Bitácora de avance semanal

Semanas 1 y 2

No se registró avance en la implementación debido a que el equipo no contaba con el conocimiento suficiente sobre los temas requeridos para el proyecto. Se utilizó este tiempo para revisar los conceptos básicos de filtros FIR y el entorno de trabajo.

Semana 3

Se investigó el funcionamiento de los filtros FIR y su implementación en Python, incluyendo el uso de scipy.signal.firwin con los parámetros especificados en el proyecto. Paralelamente se estudió el funcionamiento del ADALM-Pluto SDR, sus requisitos de software y el proceso de conexión al PC. Al finalizar la semana se logró diseñar los filtros paso bajo, paso alto y paso banda, y se dejó todo preparado para la conexión con el Pluto.

Semana 3 — Conexión con el Pluto

Se completó la conexión del ADALM-Pluto. Se presentó un problema con el driver libiio que no era reconocido por Python — se resolvió instalando los drivers oficiales de PlutoSDR e instalando manualmente el archivo libiio.dll. También se identificó que el cable USB original era solo de carga, por lo que se reemplazó por un cable de datos micro-USB.

Semana 4

Se capturó la señal multi-tono con el receptor RX del Pluto y se aplicaron los tres filtros FIR. La señal inicialmente presentaba ruido elevado, lo que se corrigió aumentando la ganancia RX de 30 a 64 dB y acercando las antenas. Se trabajó con 16384 muestras para obtener buena resolución en la FFT. Al cerrar el script aparecía un error OSError relacionado con la liberación del buffer, aunque se confirmó que es un bug conocido de pyadi-iio en Windows y no afecta los resultados.

Semana 5

No se registró avance durante esta semana por compromisos externos del equipo.

Semana 6

Se recuperó el atraso de la semana 5 y se avanzó adicionalmente hasta completar las actividades de la semana 6. Se realizó la comparación de filtros con 32, 64 y 128 taps y la validación experimental, obteniendo un error promedio menor a 0.2 dB en la mayoría de frecuencias evaluadas.

Semana 7

Se completaron todas las actividades planificadas. Con los datos obtenidos durante el proyecto y el conocimiento adquirido en la investigación, se generaron las gráficas comparativas finales y se organizó el material para la elaboración del informe técnico en Overleaf.
---
## Referencias

- Proakis, J. & Manolakis, D. — *Digital Signal Processing*
- Analog Devices — *ADALM-Pluto documentation*
- Collins, T. et al. — *PySDR: A Guide to SDR and DSP using Python*
- scipy.signal documentation: https://docs.scipy.org/doc/scipy/reference/signal.html
