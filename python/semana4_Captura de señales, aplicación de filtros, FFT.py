"""
Proyecto: Filtro Digital FIR en Python para Señales SDR
Semana 4: Captura de señal con ADALM-Pluto y aplicación de filtros FIR
Universidad Nacional de Chimborazo - Escuela de Telecomunicaciones

Especificaciones:
  - Señal TX multi-tono : 50 + 200 + 300 kHz
  - Sample rate         : 1 MHz
  - Frecuencia central  : 915 MHz
  - Filtro paso bajo    : fc = 100 kHz, 65 taps, Hamming
  - Filtro paso alto    : fc = 200 kHz, 65 taps, Hamming
  - Filtro paso banda   : fc = [150, 250] kHz, 65 taps, Hamming

Dependencias:
  pip install pyadi-iio numpy scipy matplotlib
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter
from scipy.fft import fft, fftfreq
import adi
import time

# ─────────────────────────────────────────────
# CONFIGURACIÓN MATPLOTLIB — Lineamientos IEEE
# ─────────────────────────────────────────────
matplotlib.rcParams.update({
    'font.family'    : 'serif',
    'font.size'      : 12,
    'axes.labelsize' : 13,
    'axes.titlesize' : 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi'     : 150,
    'savefig.dpi'    : 300,
    'savefig.bbox'   : 'tight',
    'lines.linewidth': 1.5,
    'grid.alpha'     : 0.3,
})

# ─────────────────────────────────────────────
# PARÁMETROS
# ─────────────────────────────────────────────
FS      = 1_000_000       # Sample rate: 1 MHz
FC      = 915_000_000     # Frecuencia central: 915 MHz
TX_GAIN = 0           # Ganancia TX (dB)
RX_GAIN = 60              # Ganancia RX (dB)
N       = 2**14           # Número de muestras
NYQUIST = FS / 2
N_TAPS  = 65
WINDOW  = "hamming"

# Frecuencias de los tonos
F1, F2, F3 = 50e3, 200e3, 300e3

# ─────────────────────────────────────────────
# 1. DISEÑO DE FILTROS FIR
# ─────────────────────────────────────────────
h_lp = firwin(N_TAPS, cutoff=100e3  / NYQUIST, window=WINDOW)
h_hp = firwin(N_TAPS, cutoff=200e3  / NYQUIST, window=WINDOW, pass_zero=False)
h_bp = firwin(N_TAPS, cutoff=[150e3 / NYQUIST, 250e3 / NYQUIST], window=WINDOW, pass_zero=False)

print("=" * 55)
print("  SEMANA 4 — Captura y filtrado con ADALM-Pluto")
print("=" * 55)
print("\n  Filtros FIR diseñados:")
print(f"    Paso bajo  : fc = 100 kHz, {N_TAPS} taps")
print(f"    Paso alto  : fc = 200 kHz, {N_TAPS} taps")
print(f"    Paso banda : fc = [150, 250] kHz, {N_TAPS} taps")

# ─────────────────────────────────────────────
# 2. GENERACIÓN DE SEÑAL TX
# ─────────────────────────────────────────────
t_vec    = np.arange(N) / FS
señal_iq = (np.exp(1j * 2 * np.pi * F1 * t_vec) +
            np.exp(1j * 2 * np.pi * F2 * t_vec) +
            np.exp(1j * 2 * np.pi * F3 * t_vec))

señal_norm = señal_iq / np.max(np.abs(señal_iq))
señal_tx   = (señal_norm * 2**14).astype(np.complex64)

# ─────────────────────────────────────────────
# 3. CONEXIÓN AL PLUTO
# ─────────────────────────────────────────────
print("\n  Conectando al ADALM-Pluto...")

try:
    sdr = adi.Pluto("ip:192.168.2.1")

    # Configurar TX
    sdr.sample_rate              = FS
    sdr.tx_rf_bandwidth          = FS
    sdr.tx_lo                    = FC
    sdr.tx_hardwaregain_chan0    = TX_GAIN
    sdr.tx_cyclic_buffer         = True

    # Configurar RX
    sdr.rx_rf_bandwidth          = FS
    sdr.rx_lo                    = FC
    sdr.rx_hardwaregain_chan0    = RX_GAIN
    sdr.rx_buffer_size           = N

    print(f"  Pluto conectado ✅")
    print(f"  TX LO : {sdr.tx_lo / 1e6:.0f} MHz | TX Gain : {sdr.tx_hardwaregain_chan0} dB")
    print(f"  RX LO : {sdr.rx_lo / 1e6:.0f} MHz | RX Gain : {sdr.rx_hardwaregain_chan0} dB")

    # ─────────────────────────────────────────
    # 4. TRANSMITIR Y CAPTURAR
    # ─────────────────────────────────────────
    print("\n  Transmitiendo señal multi-tono...")
    sdr.tx(señal_tx)

    # Esperar que el transmisor se estabilice
    time.sleep(0.5)

    # Capturar muestras RX
    print("  Capturando señal RX...")
    rx_samples = sdr.rx()
    print(f"  Muestras capturadas: {len(rx_samples)}")

    # Detener TX
    sdr.tx_destroy_buffer()
    print("  Transmisión detenida.")

except Exception as e:
    print(f"\n  Error con el Pluto: {e}")
    print("  Usando señal simulada para continuar...")
    # Señal simulada con ruido si no hay Pluto
    ruido      = 0.1 * (np.random.randn(N) + 1j * np.random.randn(N))
    rx_samples = señal_norm + ruido

# ─────────────────────────────────────────────
# 5. APLICAR FILTROS A LA SEÑAL CAPTURADA
# ─────────────────────────────────────────────
rx_real = rx_samples.real  # Trabajar con parte real

salida_lp = lfilter(h_lp, 1.0, rx_real)
salida_hp = lfilter(h_hp, 1.0, rx_real)
salida_bp = lfilter(h_bp, 1.0, rx_real)

# ─────────────────────────────────────────────
# 6. FFT DE SEÑALES
# ─────────────────────────────────────────────
def calcular_fft_dB(señal, fs):
    N     = len(señal)
    X     = fft(señal)
    freqs = fftfreq(N, d=1/fs)
    idx   = np.argsort(freqs)
    freqs = freqs[idx]
    X_dB  = 20 * np.log10(np.abs(X[idx]) / N + 1e-12)
    return freqs, X_dB

freqs_rx, Xrx_dB = calcular_fft_dB(rx_real,    FS)
freqs_lp, Xlp_dB = calcular_fft_dB(salida_lp,  FS)
freqs_hp, Xhp_dB = calcular_fft_dB(salida_hp,  FS)
freqs_bp, Xbp_dB = calcular_fft_dB(salida_bp,  FS)

# ─────────────────────────────────────────────
# 7. MÉTRICAS DE VALIDACIÓN
# ─────────────────────────────────────────────
def ganancia_en(freqs, X_dB, f):
    return X_dB[np.argmin(np.abs(freqs - f))]

print("\n  Validación — Ganancia en cada tono después del filtrado:")
print(f"  {'Filtro':<12} {'@ 50 kHz':>10} {'@ 200 kHz':>10} {'@ 300 kHz':>10}")
print(f"  {'-'*44}")
for nombre, freqs, X_dB in [('Paso bajo', freqs_lp, Xlp_dB),
                              ('Paso alto', freqs_hp, Xhp_dB),
                              ('Paso banda', freqs_bp, Xbp_dB)]:
    g1 = ganancia_en(freqs, X_dB, F1)
    g2 = ganancia_en(freqs, X_dB, F2)
    g3 = ganancia_en(freqs, X_dB, F3)
    print(f"  {nombre:<12} {g1:>9.1f} dB {g2:>9.1f} dB {g3:>9.1f} dB")

print("=" * 55)

# ─────────────────────────────────────────────
# 8. FIGURAS
# ─────────────────────────────────────────────

ESTILO_ENT = dict(color='gray',  linestyle=':',  linewidth=1.2, label='Señal capturada RX')
ESTILO_LP  = dict(color='black', linestyle='-',  linewidth=1.5, label='Salida paso bajo')
ESTILO_HP  = dict(color='black', linestyle='--', linewidth=1.5, label='Salida paso alto')
ESTILO_BP  = dict(color='black', linestyle='-.', linewidth=1.5, label='Salida paso banda')

# ── Fig. 1: Señal RX capturada ────────────────
fig1, ax = plt.subplots(figsize=(13, 4))
ax.plot(np.arange(500) / FS * 1e6, rx_real[:500],
        color='black', linestyle='-', linewidth=1.2)
ax.set_title('Señal capturada RX — Parte real (primeras 500 muestras)')
ax.set_xlabel('Tiempo (µs)')
ax.set_ylabel('Amplitud')
ax.grid(True)
fig1.suptitle('Fig. 1 — Señal I/Q recibida del ADALM-Pluto', fontsize=14, y=1.02)
fig1.tight_layout()
fig1.savefig("semana4_señal_rx.pdf")
plt.show()

# ── Fig. 2: FFT antes y después de cada filtro
fig2, axes = plt.subplots(1, 3, figsize=(18, 5))
fig2.subplots_adjust(wspace=0.35)

titulos = ['Filtro Paso Bajo (100 kHz)',
           'Filtro Paso Alto (200 kHz)',
           'Filtro Paso Banda (150–250 kHz)']
estilos = [ESTILO_LP, ESTILO_HP, ESTILO_BP]
datos   = [(freqs_lp, Xlp_dB), (freqs_hp, Xhp_dB), (freqs_bp, Xbp_dB)]

for i, (ax, titulo, estilo, (freqs, X_dB)) in enumerate(
        zip(axes, titulos, estilos, datos)):
    ax.plot(freqs_rx / 1e3, Xrx_dB, **ESTILO_ENT)
    ax.plot(freqs    / 1e3, X_dB,   **estilo)
    ax.set_xlim(-500, 500)
    ax.set_ylim(-80, 10)
    ax.set_title(titulo)
    ax.set_xlabel('Frecuencia (kHz)')
    ax.set_ylabel('Magnitud (dB)')
    ax.legend(loc='lower center')
    ax.grid(True)
    # Marcar tonos
    for f in [F1, F2, F3, -F1, -F2, -F3]:
        ax.axvline(f / 1e3, color='gray', linestyle='--', linewidth=0.7, alpha=0.6)

fig2.suptitle('Fig. 2 — Espectro FFT antes y después del filtrado FIR', fontsize=14, y=1.02)
fig2.tight_layout()
fig2.savefig("semana4_fft_filtros.pdf")
plt.show()

print("\n  PDFs guardados: semana4_señal_rx.pdf, semana4_fft_filtros.pdf")
print("  Semana 4 completada ✅")
=======================================================
  SEMANA 4 — Captura y filtrado con ADALM-Pluto
=======================================================

  Filtros FIR diseñados:
    Paso bajo  : fc = 100 kHz, 65 taps
    Paso alto  : fc = 200 kHz, 65 taps
    Paso banda : fc = [150, 250] kHz, 65 taps

  Conectando al ADALM-Pluto...
  Pluto conectado ✅
  TX LO : 915 MHz | TX Gain : 0 dB
  RX LO : 915 MHz | RX Gain : 70 dB

  Transmitiendo señal multi-tono...
  Capturando señal RX...
  Muestras capturadas: 16384
  Transmisión detenida.

  Validación — Ganancia en cada tono después del filtrado:
  Filtro         @ 50 kHz  @ 200 kHz  @ 300 kHz
  --------------------------------------------
  Paso bajo         44.6 dB     -35.4 dB     -15.7 dB
  Paso alto        -15.1 dB      38.5 dB      44.4 dB
  Paso banda       -41.2 dB      44.5 dB     -10.6 dB
=======================================================

  PDFs guardados: semana4_señal_rx.pdf, semana4_fft_filtros.pdf
  Semana 4 completada ✅
