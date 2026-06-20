"""
Proyecto: Filtro Digital FIR en Python para Señales SDR
Semana 6: Validación — verificación de atenuación y ganancia
Universidad Nacional de Chimborazo - Escuela de Telecomunicaciones

Especificaciones:
  - Señal TX multi-tono : 50 + 200 + 300 kHz
  - Sample rate         : 1 MHz
  - Frecuencia central  : 915 MHz
  - Filtros validados   : paso bajo, paso alto, paso banda (65 taps)

Métricas:
  - Ganancia en passband (dB)
  - Atenuación en stopband (dB)
  - Comparación teórico vs experimental
  - Error relativo (%)

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

# ─────────────────────────────────────────────
# 1. DISEÑO DE FILTROS
# ─────────────────────────────────────────────
h_lp = firwin(N_TAPS, cutoff=100e3  / NYQUIST, window=WINDOW)
h_hp = firwin(N_TAPS, cutoff=200e3  / NYQUIST, window=WINDOW, pass_zero=False)
h_bp = firwin(N_TAPS, cutoff=[150e3 / NYQUIST, 250e3 / NYQUIST], window=WINDOW, pass_zero=False)

# Ganancia teórica de cada filtro en cada tono
def ganancia_teorica(h, fs, freq):
    w, H = freqz(h, worN=8192, fs=fs)
    H_dB = 20 * np.log10(np.abs(H) + 1e-12)
    return H_dB[np.argmin(np.abs(w - freq))]

teorico = {
    'lp': {F1: ganancia_teorica(h_lp, FS, F1),
            F2: ganancia_teorica(h_lp, FS, F2),
            F3: ganancia_teorica(h_lp, FS, F3)},
    'hp': {F1: ganancia_teorica(h_hp, FS, F1),
            F2: ganancia_teorica(h_hp, FS, F2),
            F3: ganancia_teorica(h_hp, FS, F3)},
    'bp': {F1: ganancia_teorica(h_bp, FS, F1),
            F2: ganancia_teorica(h_bp, FS, F2),
            F3: ganancia_teorica(h_bp, FS, F3)},
}

print("=" * 60)
print("  SEMANA 6 — Validación: Atenuación y Ganancia")
print("=" * 60)
print("\n  Ganancias teóricas (dB):")
print(f"  {'Filtro':<12} {'@ 50 kHz':>10} {'@ 200 kHz':>10} {'@ 300 kHz':>10}")
print(f"  {'-'*44}")
for nombre, key in [('Paso bajo', 'lp'), ('Paso alto', 'hp'), ('Paso banda', 'bp')]:
    print(f"  {nombre:<12} {teorico[key][F1]:>10.1f} {teorico[key][F2]:>10.1f} {teorico[key][F3]:>10.1f}")

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

# ─────────────────────────────────────────────
# 5. GANANCIA EXPERIMENTAL VÍA FFT
# ─────────────────────────────────────────────
def ganancia_experimental(señal, fs, freq):
    X     = fft(señal)
    freqs = fftfreq(len(señal), d=1/fs)
    X_dB  = 20 * np.log10(np.abs(X) / len(señal) + 1e-12)
    return X_dB[np.argmin(np.abs(freqs - freq))]

# Ganancia de referencia (sin filtro)
ref = {f: ganancia_experimental(rx_real, FS, f) for f in [F1, F2, F3]}

experimental = {
    'lp': {f: ganancia_experimental(salida_lp, FS, f) - ref[f] for f in [F1, F2, F3]},
    'hp': {f: ganancia_experimental(salida_hp, FS, f) - ref[f] for f in [F1, F2, F3]},
    'bp': {f: ganancia_experimental(salida_bp, FS, f) - ref[f] for f in [F1, F2, F3]},
}

# ─────────────────────────────────────────────
# 6. TABLA COMPARATIVA TEÓRICO VS EXPERIMENTAL
# ─────────────────────────────────────────────
print("\n  Comparación Teórico vs Experimental:")
print(f"\n  {'Filtro':<11} {'Tono':>8} {'Teórico':>10} {'Experim.':>10} {'Error':>8}")
print(f"  {'-'*52}")

resultados = []
for nombre, key in [('Paso bajo', 'lp'), ('Paso alto', 'hp'), ('Paso banda', 'bp')]:
    for f, lbl in [(F1, '50 kHz'), (F2, '200 kHz'), (F3, '300 kHz')]:
        teo = teorico[key][f]
        exp = experimental[key][f]
        err = abs(exp - teo)
        print(f"  {nombre:<11} {lbl:>8} {teo:>9.1f} dB {exp:>9.1f} dB {err:>6.1f} dB")
        resultados.append((nombre, lbl, teo, exp, err))

print("=" * 60)

# ─────────────────────────────────────────────
# 7. FFT DE SEÑALES
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

# ─────────────────────────────────────────────
# 8. FIGURAS
# ─────────────────────────────────────────────

COLORES = {'lp': 'steelblue', 'hp': 'darkorange', 'bp': 'green'}

# ── Fig. 1: FFT señal capturada + 3 filtros ───
fig1, axes = plt.subplots(1, 3, figsize=(18, 5))
fig1.subplots_adjust(wspace=0.35)

configs = [
    ('Paso Bajo (100 kHz)',     freqs_lp, Xlp, 'steelblue',  '-'),
    ('Paso Alto (200 kHz)',     freqs_hp, Xhp, 'darkorange', '--'),
    ('Paso Banda (150–250 kHz)',freqs_bp, Xbp, 'green',      '-.'),
]

for i, (titulo, freqs_f, Xf, color, ls) in enumerate(configs):
    axes[i].plot(freqs_rx / 1e3, Xrx, color='gray', linestyle=':', linewidth=1.2, label='RX capturada')
    axes[i].plot(freqs_f  / 1e3, Xf,  color=color,  linestyle=ls,  linewidth=1.8, label=titulo)
    for f in [F1, F2, F3, -F1, -F2, -F3]:
        axes[i].axvline(f / 1e3, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)
    axes[i].set_xlim(-500, 500)
    axes[i].set_ylim(-80, 10)
    axes[i].set_title(titulo)
    axes[i].set_xlabel('Frecuencia (kHz)')
    axes[i].set_ylabel('Magnitud (dB)')
    axes[i].legend(loc='lower center')
    axes[i].grid(True)

fig1.suptitle('Fig. 1 — Validación: Espectro antes y después del filtrado', fontsize=14, y=1.02)
fig1.tight_layout()
fig1.savefig("semana6_validacion_fft.pdf")
plt.show()

# ── Fig. 2: Tabla comparativa teórico vs exp ──
fig2, ax = plt.subplots(figsize=(14, 5))
ax.axis('off')

col_labels = ['Filtro', 'Tono', 'Teórico (dB)', 'Experimental (dB)', 'Error (dB)']
cell_data  = [[r[0], r[1], f"{r[2]:.1f}", f"{r[3]:.1f}", f"{r[4]:.1f}"] for r in resultados]

tabla = ax.table(
    cellText=cell_data,
    colLabels=col_labels,
    loc='center',
    cellLoc='center'
)
tabla.auto_set_font_size(False)
tabla.set_fontsize(11)
tabla.scale(1.2, 2.0)

# Color de encabezado
for j in range(len(col_labels)):
    tabla[0, j].set_facecolor('#2c3e50')
    tabla[0, j].set_text_props(color='white', fontweight='bold')

fig2.suptitle('Fig. 2 — Comparación Teórico vs Experimental', fontsize=14)
fig2.tight_layout()
fig2.savefig("semana6_tabla_comparativa.pdf")
plt.show()

# ── Fig. 3: Barras de ganancia por tono ───────
fig3, axes = plt.subplots(1, 3, figsize=(15, 5))
fig3.subplots_adjust(wspace=0.4)

tonos  = ['50 kHz', '200 kHz', '300 kHz']
freqs_list = [F1, F2, F3]
filtros_nombres = ['Paso bajo', 'Paso alto', 'Paso banda']
filtros_keys    = ['lp', 'hp', 'bp']
colores_barra   = ['steelblue', 'darkorange', 'green']

x = np.arange(len(tonos))
ancho = 0.35

for i, (nombre, key, color) in enumerate(zip(filtros_nombres, filtros_keys, colores_barra)):
    teo_vals = [teorico[key][f]     for f in freqs_list]
    exp_vals = [experimental[key][f] for f in freqs_list]

    axes[i].bar(x - ancho/2, teo_vals, ancho, label='Teórico',
                color=color, alpha=0.5, edgecolor='black')
    axes[i].bar(x + ancho/2, exp_vals, ancho, label='Experimental',
                color=color, alpha=1.0, edgecolor='black', hatch='//')
    axes[i].axhline(-40, color='red',  linewidth=0.8, linestyle='--', label='−40 dB')
    axes[i].axhline(0,   color='gray', linewidth=0.8, linestyle=':')
    axes[i].set_xticks(x)
    axes[i].set_xticklabels(tonos)
    axes[i].set_title(f'Filtro {nombre}')
    axes[i].set_ylabel('Ganancia (dB)')
    axes[i].set_ylim(-90, 10)
    axes[i].legend(fontsize=9)
    axes[i].grid(True, axis='y')

fig3.suptitle('Fig. 3 — Ganancia teórica vs experimental por tono y filtro', fontsize=14, y=1.02)
fig3.tight_layout()
fig3.savefig("semana6_barras_ganancia.pdf")
plt.show()

print("\n  PDFs guardados:")
print("    semana6_validacion_fft.pdf")
print("    semana6_tabla_comparativa.pdf")
print("    semana6_barras_ganancia.pdf")
print("\n  Semana 6 completada ✅")
print("=" * 60)

============================================================
  SEMANA 6 — Validación: Atenuación y Ganancia
============================================================

  Ganancias teóricas (dB):
  Filtro         @ 50 kHz  @ 200 kHz  @ 300 kHz
  --------------------------------------------
  Paso bajo          -0.0      -92.3      -59.5
  Paso alto         -60.3       -6.0        0.0
  Paso banda        -93.2       -0.0      -55.6

  Conectando al ADALM-Pluto...
  Pluto conectado ✅
  Muestras capturadas: 16384

  Comparación Teórico vs Experimental:

  Filtro          Tono    Teórico   Experim.    Error
  ----------------------------------------------------
  Paso bajo     50 kHz      -0.0 dB      -0.0 dB    0.0 dB
  Paso bajo    200 kHz     -92.3 dB     -84.2 dB    8.1 dB
  Paso bajo    300 kHz     -59.5 dB     -59.4 dB    0.1 dB
  Paso alto     50 kHz     -60.3 dB     -60.1 dB    0.2 dB
  Paso alto    200 kHz      -6.0 dB      -6.0 dB    0.0 dB
  Paso alto    300 kHz       0.0 dB      -0.0 dB    0.0 dB
  Paso banda    50 kHz     -93.2 dB     -93.3 dB    0.1 dB
  Paso banda   200 kHz      -0.0 dB      -0.0 dB    0.0 dB
  Paso banda   300 kHz     -55.6 dB     -55.8 dB    0.2 dB
============================================================

  PDFs guardados:
    semana6_validacion_fft.pdf
    semana6_tabla_comparativa.pdf
    semana6_barras_ganancia.pdf

  Semana 6 completada ✅
============================================================
