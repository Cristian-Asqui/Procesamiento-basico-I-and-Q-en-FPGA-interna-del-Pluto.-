"""
Proyecto: Filtro Digital FIR en Python para Señales SDR
Semana 7: Análisis de resultados y gráficas comparativas finales
Universidad Nacional de Chimborazo - Escuela de Telecomunicaciones

Especificaciones:
  - Señal TX multi-tono : 50 + 200 + 300 kHz
  - Sample rate         : 1 MHz
  - Frecuencia central  : 915 MHz
  - Filtros             : paso bajo, paso alto, paso banda (65 taps)
  - Comparación         : 32, 64, 128 taps

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
N_TAPS  = 65
WINDOW  = "hamming"
F1, F2, F3 = 50e3, 200e3, 300e3
TAPS_LIST  = [33, 65, 129]

COLORES = {
    'lp': 'steelblue',
    'hp': 'darkorange',
    'bp': 'green',
}

# ─────────────────────────────────────────────
# 1. DISEÑO DE FILTROS
# ─────────────────────────────────────────────
h_lp = firwin(N_TAPS, cutoff=100e3  / NYQUIST, window=WINDOW)
h_hp = firwin(N_TAPS, cutoff=200e3  / NYQUIST, window=WINDOW, pass_zero=False)
h_bp = firwin(N_TAPS, cutoff=[150e3 / NYQUIST, 250e3 / NYQUIST], window=WINDOW, pass_zero=False)

filtros_tap = {}
for n in TAPS_LIST:
    h = firwin(n, cutoff=100e3 / NYQUIST, window=WINDOW)
    w, H = freqz(h, worN=8192, fs=FS)
    filtros_tap[n] = {'h': h, 'w': w, 'H_dB': 20 * np.log10(np.abs(H) + 1e-12)}

w_lp, H_lp = freqz(h_lp, worN=8192, fs=FS)
w_hp, H_hp = freqz(h_hp, worN=8192, fs=FS)
w_bp, H_bp = freqz(h_bp, worN=8192, fs=FS)

H_lp_dB = 20 * np.log10(np.abs(H_lp) + 1e-12)
H_hp_dB = 20 * np.log10(np.abs(H_hp) + 1e-12)
H_bp_dB = 20 * np.log10(np.abs(H_bp) + 1e-12)

print("=" * 60)
print("  SEMANA 7 — Análisis de resultados y gráficas finales")
print("=" * 60)

# ─────────────────────────────────────────────
# 2. SEÑAL TX
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
    rx_samples = sdr.rx()
    sdr.tx_destroy_buffer()
    print(f"  Muestras capturadas: {len(rx_samples)}")

except Exception as e:
    print(f"  Error Pluto: {e} — usando señal simulada")
    ruido      = 0.05 * (np.random.randn(N) + 1j * np.random.randn(N))
    rx_samples = señal_norm + ruido

rx_real = rx_samples.real

# ─────────────────────────────────────────────
# 4. APLICAR FILTROS
# ─────────────────────────────────────────────
salida_lp = lfilter(h_lp, 1.0, rx_real)
salida_hp = lfilter(h_hp, 1.0, rx_real)
salida_bp = lfilter(h_bp, 1.0, rx_real)

salidas_tap = {}
tiempos_tap = {}
for n in TAPS_LIST:
    t0 = time.perf_counter()
    salidas_tap[n] = lfilter(filtros_tap[n]['h'], 1.0, rx_real)
    tiempos_tap[n] = (time.perf_counter() - t0) * 1000

# ─────────────────────────────────────────────
# 5. FFT
# ─────────────────────────────────────────────
def calcular_fft_dB(señal, fs):
    X     = fft(señal)
    freqs = fftfreq(len(señal), d=1/fs)
    idx   = np.argsort(freqs)
    return freqs[idx], 20 * np.log10(np.abs(X[idx]) / len(señal) + 1e-12)

freqs_rx, Xrx = calcular_fft_dB(rx_real,   FS)
freqs_lp, Xlp = calcular_fft_dB(salida_lp, FS)
freqs_hp, Xhp = calcular_fft_dB(salida_hp, FS)
freqs_bp, Xbp = calcular_fft_dB(salida_bp, FS)
ffts_tap = {n: calcular_fft_dB(salidas_tap[n], FS) for n in TAPS_LIST}

# ─────────────────────────────────────────────
# 6. FIGURAS FINALES
# ─────────────────────────────────────────────

# ── Fig. 1: Respuestas en frecuencia — 3 filtros
fig1, ax = plt.subplots(figsize=(13, 5))
ax.plot(w_lp / 1e3, H_lp_dB, color=COLORES['lp'], linestyle='-',  linewidth=1.8, label='Paso bajo (100 kHz)')
ax.plot(w_hp / 1e3, H_hp_dB, color=COLORES['hp'], linestyle='--', linewidth=1.8, label='Paso alto (200 kHz)')
ax.plot(w_bp / 1e3, H_bp_dB, color=COLORES['bp'], linestyle='-.', linewidth=1.8, label='Paso banda (150–250 kHz)')
ax.axhline(-40, color='gray', linewidth=0.8, linestyle=':', label='−40 dB objetivo')
ax.axhline(-1,  color='gray', linewidth=0.8, linestyle='--', label='−1 dB ripple')
for f, lbl in [(50, '50k'), (200, '200k'), (300, '300k')]:
    ax.axvline(f, color='gray', linewidth=0.7, linestyle='--', alpha=0.5)
    ax.text(f + 3, -85, lbl, fontsize=9, color='gray')
ax.set_xlim(0, 500)
ax.set_ylim(-90, 5)
ax.set_title('Respuesta en Frecuencia — Filtros FIR diseñados (65 taps, Hamming)')
ax.set_xlabel('Frecuencia (kHz)')
ax.set_ylabel('$|H(f)|$ (dB)')
ax.legend(loc='lower right')
ax.grid(True)
fig1.suptitle('Fig. 1 — Respuesta en frecuencia de los tres filtros FIR', fontsize=14, y=1.02)
fig1.tight_layout()
fig1.savefig("semana7_respuesta_tres_filtros.pdf")
plt.show()

# ── Fig. 2: FFT señal real — 3 filtros juntos
fig2, ax = plt.subplots(figsize=(13, 5))
ax.plot(freqs_rx / 1e3, Xrx, color='gray',          linestyle=':',  linewidth=1.2, label='Señal RX capturada', alpha=0.8)
ax.plot(freqs_lp / 1e3, Xlp, color=COLORES['lp'],   linestyle='-',  linewidth=1.8, label='Salida paso bajo')
ax.plot(freqs_hp / 1e3, Xhp, color=COLORES['hp'],   linestyle='--', linewidth=1.8, label='Salida paso alto')
ax.plot(freqs_bp / 1e3, Xbp, color=COLORES['bp'],   linestyle='-.', linewidth=1.8, label='Salida paso banda')
for f in [F1, F2, F3, -F1, -F2, -F3]:
    ax.axvline(f / 1e3, color='gray', linewidth=0.7, linestyle='--', alpha=0.4)
ax.set_xlim(-500, 500)
ax.set_ylim(-80, 10)
ax.set_title('Espectro FFT — Señal real del Pluto antes y después del filtrado')
ax.set_xlabel('Frecuencia (kHz)')
ax.set_ylabel('Magnitud (dB)')
ax.legend(loc='lower center', ncol=2)
ax.grid(True)
fig2.suptitle('Fig. 2 — Comparación espectral: señal capturada vs señales filtradas', fontsize=14, y=1.02)
fig2.tight_layout()
fig2.savefig("semana7_comparacion_espectral.pdf")
plt.show()

# ── Fig. 3: Trade-offs 32/64/128 taps
fig3, axes = plt.subplots(1, 2, figsize=(13, 5))
fig3.subplots_adjust(wspace=0.35)

ESTILOS_TAP = {
    TAPS_LIST[0]: dict(color='steelblue',  linestyle='-',  linewidth=1.8, label=f'{TAPS_LIST[0]-1} taps'),
    TAPS_LIST[1]: dict(color='darkorange', linestyle='--', linewidth=1.8, label=f'{TAPS_LIST[1]-1} taps'),
    TAPS_LIST[2]: dict(color='green',      linestyle=':',  linewidth=2.2, label=f'{TAPS_LIST[2]-1} taps'),
}

# Respuestas en frecuencia comparadas
for n in TAPS_LIST:
    axes[0].plot(filtros_tap[n]['w'] / 1e3, filtros_tap[n]['H_dB'], **ESTILOS_TAP[n])
axes[0].axhline(-40, color='gray', linewidth=0.8, linestyle='-.', label='−40 dB')
axes[0].set_xlim(0, 300)
axes[0].set_ylim(-90, 5)
axes[0].set_title('Respuesta en frecuencia — 32, 64, 128 taps')
axes[0].set_xlabel('Frecuencia (kHz)')
axes[0].set_ylabel('$|H(f)|$ (dB)')
axes[0].legend()
axes[0].grid(True)

# FFT señal filtrada comparada
for n in TAPS_LIST:
    freqs_f, Xf = ffts_tap[n]
    axes[1].plot(freqs_f / 1e3, Xf, **ESTILOS_TAP[n])
axes[1].plot(freqs_rx / 1e3, Xrx, color='gray', linestyle=':', linewidth=1.0, label='RX capturada', alpha=0.6)
axes[1].set_xlim(-500, 500)
axes[1].set_ylim(-80, 10)
axes[1].set_title('FFT señal filtrada — 32, 64, 128 taps')
axes[1].set_xlabel('Frecuencia (kHz)')
axes[1].set_ylabel('Magnitud (dB)')
axes[1].legend()
axes[1].grid(True)

fig3.suptitle('Fig. 3 — Trade-off: orden del filtro vs calidad de filtrado', fontsize=14, y=1.02)
fig3.tight_layout()
fig3.savefig("semana7_tradeoff_taps.pdf")
plt.show()

# ── Fig. 4: Tabla resumen final del proyecto
fig4, ax = plt.subplots(figsize=(22, 6))
ax.axis('off')

col_labels = ['Semana', 'Actividad', 'Resultado principal', 'Estado']
cell_data = [
    ['1', 'Filtro paso bajo',       'fc=100 kHz, atten=59 dB, ripple=0.02 dB', 'Completado'],
    ['2', 'Paso alto y paso banda', 'fc=200/[150-250] kHz, 65 taps',            'Completado'],
    ['3', 'Transmision con Pluto',  'TX 915 MHz, 50+200+300 kHz',               'Completado'],
    ['4', 'Captura y filtrado',     'RX capturado, 3 filtros aplicados',        'Completado'],
    ['5', 'Comp. 32/64/128 taps',   'trans=14.8 kHz, ripple=0.03 dB',          'Completado'],
    ['6', 'Validacion',             'Error promedio < 0.2 dB',                  'Completado'],
    ['7', 'Analisis final',         'Graficas comparativas generadas',           'Completado'],
]

tabla = ax.table(
    cellText=cell_data,
    colLabels=col_labels,
    loc='center',
    cellLoc='center'
)
tabla.auto_set_font_size(False)
tabla.set_fontsize(11)
tabla.scale(1.0, 2.2)
tabla.auto_set_column_width([0, 1, 2, 3])

for j in range(len(col_labels)):
    tabla[0, j].set_facecolor('#2c3e50')
    tabla[0, j].set_text_props(color='white', fontweight='bold')
for i in range(1, len(cell_data) + 1):
    tabla[i, 3].set_facecolor('#d5f5e3')

fig4.suptitle('Fig. 4 — Tabla resumen del proyecto: Filtros FIR con ADALM-Pluto', fontsize=14)
fig4.tight_layout()
fig4.savefig("semana7_tabla_resumen.pdf")
plt.show()

print("\n  PDFs guardados:")
print("    semana7_respuesta_tres_filtros.pdf")
print("    semana7_comparacion_espectral.pdf")
print("    semana7_tradeoff_taps.pdf")
print("    semana7_tabla_resumen.pdf")
print("\n  Semana 7 completada ✅")
print("=" * 60)
============================================================
  SEMANA 7 — Análisis de resultados y gráficas finales
============================================================

  Conectando al ADALM-Pluto...
  Pluto conectado ✅
  Muestras capturadas: 16384

  PDFs guardados:
    semana7_respuesta_tres_filtros.pdf
    semana7_comparacion_espectral.pdf
    semana7_tradeoff_taps.pdf
    semana7_tabla_resumen.pdf

  Semana 7 completada ✅
============================================================
