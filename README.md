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

## Referencias

- Proakis, J. & Manolakis, D. — *Digital Signal Processing*
- Analog Devices — *ADALM-Pluto documentation*
- Collins, T. et al. — *PySDR: A Guide to SDR and DSP using Python*
- scipy.signal documentation: https://docs.scipy.org/doc/scipy/reference/signal.html
