"""
Proyecto: Filtro Digital FIR en Python para Señales SDR
Semana 5: Comparación de filtros FIR con 32, 64 y 128 taps
Universidad Nacional de Chimborazo - Escuela de Telecomunicaciones

Especificaciones:
  - Señal TX multi-tono : 50 + 200 + 300 kHz
  - Sample rate         : 1 MHz
  - Frecuencia central  : 915 MHz
  - Filtros comparados  : 32, 64, 128 taps (paso bajo fc=100 kHz)
  - Ventana             : Hamming

Métricas evaluadas:
  - Atenuación stopband (dB)
  - Ripple passband (dB)
  - Ancho de transición (kHz)
  - Tiempo de procesamiento (ms)

Dependencias:
  pip install pyadi-iio numpy scipy matplotlib
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import firwin, freqz, lfilter
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
FS      = 1_000_000
FC      = 915_000_000
TX_GAIN = 0
RX_GAIN = 64
N       = 2**14
NYQUIST = FS / 2
WINDOW  = "hamming"
FC_LP   = 100e3           # Frecuencia de corte paso bajo

TAPS_LIST = [33, 65, 129]  # Impares equivalentes a 32, 64, 128

F1, F2, F3 = 50e3, 200e3, 300e3

# ─────────────────────────────────────────────
# 1. DISEÑO DE FILTROS PARA CADA ORDEN
# ─────────────────────────────────────────────
filtros = {}
for n in TAPS_LIST:
    h = firwin(n, cutoff=FC_LP / NYQUIST, window=WINDOW)
    w, H = freqz(h, worN=8192, fs=FS)
    H_dB = 20 * np.log10(np.abs(H) + 1e-12)
    filtros[n] = {'h': h, 'w': w, 'H_dB': H_dB}

print("=" * 60)
print("  SEMANA 5 — Comparación de filtros FIR: 32, 64, 128 taps")
print("=" * 60)

# ─────────────────────────────────────────────
# 2. MÉTRICAS TEÓRICAS
# ─────────────────────────────────────────────
def atenuacion_stopband(w, H_dB, f_stop):
    idx = np.argmin(np.abs(w - f_stop))
    return -H_dB[idx]

def ripple_passband(w, H_dB, f_pass):
    mask = w <= f_pass
    return np.max(H_dB[mask]) - np.min(H_dB[mask])

def ancho_transicion(w, H_dB, f_pass, f_stop):
    # Frecuencia donde cae a -3 dB y -40 dB
    idx_3  = np.argmin(np.abs(H_dB + 3))
    idx_40 = np.argmin(np.abs(H_dB + 40))
    return abs(w[idx_40] - w[idx_3]) / 1e3

print(f"\n  {'Taps':<8} {'Atten. SB (dB)':>16} {'Ripple PB (dB)':>16} {'Trans. (kHz)':>14}")
print(f"  {'-'*56}")
metricas = {}
for n in TAPS_LIST:
    d    = filtros[n]
    att  = atenuacion_stopband(d['w'], d['H_dB'], 200e3)
    rip  = ripple_passband(d['w'], d['H_dB'], 80e3)
    trans = ancho_transicion(d['w'], d['H_dB'], FC_LP, 200e3)
    metricas[n] = {'att': att, 'rip': rip, 'trans': trans}
    label = n - 1  # mostrar como 32, 64, 128
    print(f"  {label:<8} {att:>16.1f} {rip:>16.2f} {trans:>14.1f}")

# ─────────────────────────────────────────────
# 3. CONEXIÓN AL PLUTO Y CAPTURA
# ─────────────────────────────────────────────
t_vec    = np.arange(N) / FS
señal_iq = (np.exp(1j * 2 * np.pi * F1 * t_vec) +
            np.exp(1j * 2 * np.pi * F2 * t_vec) +
            np.exp(1j * 2 * np.pi * F3 * t_vec))
señal_norm = señal_iq / np.max(np.abs(señal_iq))
señal_tx   = (señal_norm * 2**14).astype(np.complex64)

print(f"\n  Conectando al ADALM-Pluto...")

try:
    sdr = adi.Pluto("ip:192.168.2.1")
    sdr.sample_rate           = FS
    sdr.tx_rf_bandwidth       = FS
    sdr.tx_lo                 = FC
    sdr.tx_hardwaregain_chan0 = TX_GAIN
    sdr.tx_cyclic_buffer      = True
    sdr.rx_rf_bandwidth       = FS
    sdr.rx_lo                 = FC
    sdr.rx_hardwaregain_chan0 = RX_GAIN
    sdr.rx_buffer_size        = N

    print(f"  Pluto conectado ✅")
    sdr.tx(señal_tx)
    time.sleep(0.5)
    print(f"  Capturando señal RX...")
    rx_samples = sdr.rx()
    sdr.tx_destroy_buffer()
    print(f"  Muestras capturadas: {len(rx_samples)}")

except Exception as e:
    print(f"  Error Pluto: {e} — usando señal simulada")
    ruido      = 0.05 * (np.random.randn(N) + 1j * np.random.randn(N))
    rx_samples = señal_norm + ruido

rx_real = rx_samples.real

# ─────────────────────────────────────────────
# 4. APLICAR FILTROS Y MEDIR TIEMPO
# ─────────────────────────────────────────────
print(f"\n  Tiempo de procesamiento:")
salidas = {}
tiempos = {}
for n in TAPS_LIST:
    t0 = time.perf_counter()
    salidas[n] = lfilter(filtros[n]['h'], 1.0, rx_real)
    t1 = time.perf_counter()
    tiempos[n] = (t1 - t0) * 1000
    label = n - 1
    print(f"    {label} taps : {tiempos[n]:.3f} ms")

# ─────────────────────────────────────────────
# 5. FFT DE SALIDAS
# ─────────────────────────────────────────────
def calcular_fft_dB(señal, fs):
    X     = fft(señal)
    freqs = fftfreq(len(señal), d=1/fs)
    idx   = np.argsort(freqs)
    return freqs[idx], 20 * np.log10(np.abs(X[idx]) / len(señal) + 1e-12)

freqs_rx, Xrx_dB = calcular_fft_dB(rx_real, FS)
ffts = {}
for n in TAPS_LIST:
    ffts[n] = calcular_fft_dB(salidas[n], FS)

# ─────────────────────────────────────────────
# 6. FIGURAS
# ─────────────────────────────────────────────

ESTILOS = {
    TAPS_LIST[0]: dict(color='steelblue', linestyle='-',  linewidth=1.8, label=f'{TAPS_LIST[0]-1} taps'),
    TAPS_LIST[1]: dict(color='darkorange', linestyle='--', linewidth=1.8, label=f'{TAPS_LIST[1]-1} taps'),
    TAPS_LIST[2]: dict(color='green',     linestyle=':',  linewidth=2.2, label=f'{TAPS_LIST[2]-1} taps'),
}

# ── Fig. 1: Respuestas en frecuencia teóricas ─
fig1, ax = plt.subplots(figsize=(13, 5))
for n in TAPS_LIST:
    ax.plot(filtros[n]['w'] / 1e3, filtros[n]['H_dB'], **ESTILOS[n])
ax.axhline(-40, color='gray', linewidth=0.8, linestyle='-.', label='−40 dB objetivo')
ax.axhline(-1,  color='gray', linewidth=0.8, linestyle=':',  label='−1 dB ripple')
ax.axvline(FC_LP / 1e3, color='black', linewidth=0.8, linestyle='-.', alpha=0.5)
ax.set_xlim(0, 500)
ax.set_ylim(-90, 5)
ax.set_title('Respuesta en Frecuencia — Comparación 32, 64, 128 taps')
ax.set_xlabel('Frecuencia (kHz)')
ax.set_ylabel('$|H(f)|$ (dB)')
ax.legend()
ax.grid(True)
fig1.suptitle('Fig. 1 — Efecto del orden del filtro FIR en la respuesta en frecuencia', fontsize=14, y=1.02)
fig1.tight_layout()
fig1.savefig("semana5_respuesta_frecuencia.pdf")
plt.show()

# ── Fig. 2: FFT señal capturada con cada filtro
fig2, axes = plt.subplots(1, 3, figsize=(18, 5))
fig2.subplots_adjust(wspace=0.35)

for i, n in enumerate(TAPS_LIST):
    freqs_f, Xf_dB = ffts[n]
    axes[i].plot(freqs_rx / 1e3, Xrx_dB,  color='gray', linestyle=':', linewidth=1.2, label='RX capturada')
    axes[i].plot(freqs_f  / 1e3, Xf_dB,   **ESTILOS[n])
    axes[i].set_xlim(-500, 500)
    axes[i].set_ylim(-80, 10)
    axes[i].set_title(f'Paso Bajo — {n-1} taps')
    axes[i].set_xlabel('Frecuencia (kHz)')
    axes[i].set_ylabel('Magnitud (dB)')
    axes[i].legend()
    axes[i].grid(True)
    for f in [F1, F2, F3, -F1, -F2, -F3]:
        axes[i].axvline(f / 1e3, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)

fig2.suptitle('Fig. 2 — FFT de señal filtrada con diferentes órdenes', fontsize=14, y=1.02)
fig2.tight_layout()
fig2.savefig("semana5_fft_comparacion.pdf")
plt.show()

# ── Fig. 3: Tabla de métricas comparativas ────
fig3, ax = plt.subplots(figsize=(10, 3))
ax.axis('off')
labels = ['32 taps', '64 taps', '128 taps']
datos_tabla = [
    [f"{metricas[n]['att']:.1f} dB",
     f"{metricas[n]['rip']:.2f} dB",
     f"{metricas[n]['trans']:.1f} kHz",
     f"{tiempos[n]:.3f} ms"]
    for n in TAPS_LIST
]
tabla = ax.table(
    cellText=datos_tabla,
    rowLabels=labels,
    colLabels=['Atenuación SB', 'Ripple PB', 'Ancho transición', 'Tiempo proc.'],
    loc='center',
    cellLoc='center'
)
tabla.auto_set_font_size(False)
tabla.set_fontsize(12)
tabla.scale(1.2, 2.0)
fig3.suptitle('Fig. 3 — Tabla comparativa: métricas vs orden del filtro FIR', fontsize=14)
fig3.tight_layout()
fig3.savefig("semana5_tabla_metricas.pdf")
plt.show()

print("\n  PDFs guardados:")
print("    semana5_respuesta_frecuencia.pdf")
print("    semana5_fft_comparacion.pdf")
print("    semana5_tabla_metricas.pdf")
print("\n  Semana 5 completada ✅")
print("=" * 60)


============================================================
  SEMANA 5 — Comparación de filtros FIR: 32, 64, 128 taps
============================================================

  Taps       Atten. SB (dB)   Ripple PB (dB)   Trans. (kHz)
  --------------------------------------------------------
  32                   59.0             1.84           59.0
  64                   92.3             0.28           29.6
  128                  67.1             0.03           14.8

  Conectando al ADALM-Pluto...
  Pluto conectado ✅
  Capturando señal RX...
  Muestras capturadas: 16384

  Tiempo de procesamiento:
    32 taps : 0.329 ms
    64 taps : 0.328 ms
    128 taps : 0.468 ms

  PDFs guardados:
    semana5_respuesta_frecuencia.pdf
    semana5_fft_comparacion.pdf
    semana5_tabla_metricas.pdf

  Semana 5 completada ✅
============================================================
