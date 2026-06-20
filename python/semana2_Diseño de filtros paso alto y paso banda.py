"""
Proyecto: Filtro Digital FIR en Python para Señales SDR
Semana 2: Diseño de Filtros FIR - Paso Alto y Paso Banda
Universidad Nacional de Chimborazo - Escuela de Telecomunicaciones

Especificaciones:
  - Paso alto : fc = 200 kHz, 65 taps, ventana Hamming
  - Paso banda : fc = [150, 250] kHz, 65 taps, ventana Hamming
  - Fs = 1 MHz

Señales de prueba:
  - 50 kHz  → ATENUADO por paso alto
  - 300 kHz → PASA por paso alto
  - 200 kHz → PASA por paso banda
  - Multi-tono: 50 + 200 + 300 kHz

Dependencias:
  pip install numpy scipy matplotlib
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import firwin, freqz, lfilter
from scipy.fft import fft, fftfreq

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
# PARÁMETROS GLOBALES
# ─────────────────────────────────────────────
FS      = 1_000_000
N_TAPS  = 65
WINDOW  = "hamming"
T       = 1e-3
N       = int(FS * T)
t       = np.arange(N) / FS
NYQUIST = FS / 2

FC_HP = 200e3
FC_BP = [150e3, 250e3]

# ─────────────────────────────────────────────
# 1. DISEÑO DE FILTROS
# ─────────────────────────────────────────────
h_hp = firwin(N_TAPS, cutoff=FC_HP / NYQUIST, window=WINDOW, pass_zero=False)
h_bp = firwin(N_TAPS, cutoff=[fc / NYQUIST for fc in FC_BP], window=WINDOW, pass_zero=False)

w_hp, H_hp = freqz(h_hp, worN=8192, fs=FS)
w_bp, H_bp = freqz(h_bp, worN=8192, fs=FS)

H_hp_dB = 20 * np.log10(np.abs(H_hp) + 1e-12)
H_bp_dB = 20 * np.log10(np.abs(H_bp) + 1e-12)

# ─────────────────────────────────────────────
# 2. SEÑALES DE PRUEBA
# ─────────────────────────────────────────────
tono_50k  = np.cos(2 * np.pi * 50e3  * t)
tono_200k = np.cos(2 * np.pi * 200e3 * t)
tono_300k = np.cos(2 * np.pi * 300e3 * t)
multitono = tono_50k + tono_200k + tono_300k

salida_hp = lfilter(h_hp, 1.0, multitono)
salida_bp = lfilter(h_bp, 1.0, multitono)

# ─────────────────────────────────────────────
# 3. FFT
# ─────────────────────────────────────────────
def calcular_fft(señal, fs):
    X     = fft(señal)
    freqs = fftfreq(len(señal), d=1/fs)
    idx   = freqs >= 0
    return freqs[idx], 2 * np.abs(X[idx]) / len(señal)

freqs_mt,  Xmt     = calcular_fft(multitono, FS)
freqs_hpo, Xhp_out = calcular_fft(salida_hp, FS)
freqs_bpo, Xbp_out = calcular_fft(salida_bp, FS)

# ─────────────────────────────────────────────
# 4. MÉTRICAS
# ─────────────────────────────────────────────
def ganancia_en(w, H_dB, f):
    return H_dB[np.argmin(np.abs(w - f))]

def ripple(w, H_dB, f1, f2):
    mask = (w >= f1) & (w <= f2)
    return np.max(H_dB[mask]) - np.min(H_dB[mask])

print("=" * 55)
print("  SEMANA 2 — Filtros FIR: Paso Alto y Paso Banda")
print("=" * 55)
print(f"\n[Paso Alto] fc={FC_HP/1e3:.0f} kHz | {N_TAPS} taps | {WINDOW}")
print(f"  @ 50 kHz  (stopband) : {ganancia_en(w_hp, H_hp_dB, 50e3):.1f} dB")
print(f"  @ 300 kHz (passband) : {ganancia_en(w_hp, H_hp_dB, 300e3):.1f} dB")
print(f"  Ripple passband      : {ripple(w_hp, H_hp_dB, 250e3, 450e3):.2f} dB")
print(f"\n[Paso Banda] fc=[{FC_BP[0]/1e3:.0f},{FC_BP[1]/1e3:.0f}] kHz | {N_TAPS} taps | {WINDOW}")
print(f"  @ 50 kHz  (stopband) : {ganancia_en(w_bp, H_bp_dB, 50e3):.1f} dB")
print(f"  @ 200 kHz (passband) : {ganancia_en(w_bp, H_bp_dB, 200e3):.1f} dB")
print(f"  @ 300 kHz (stopband) : {ganancia_en(w_bp, H_bp_dB, 300e3):.1f} dB")
print(f"  Ripple passband      : {ripple(w_bp, H_bp_dB, FC_BP[0], FC_BP[1]):.2f} dB")
print("=" * 55)

# ─────────────────────────────────────────────
# 5. FIGURAS — cada una en ventana separada
# ─────────────────────────────────────────────

ESTILO_HP     = dict(color='black', linestyle='-',  linewidth=1.5, label='Paso alto')
ESTILO_BP     = dict(color='black', linestyle='--', linewidth=1.5, label='Paso banda')
ESTILO_ENT    = dict(color='gray',  linestyle=':',  linewidth=1.2, label='Entrada (multi-tono)')
ESTILO_SAL_HP = dict(color='black', linestyle='-',  linewidth=1.5, label='Salida paso alto')
ESTILO_SAL_BP = dict(color='black', linestyle='--', linewidth=1.5, label='Salida paso banda')

# ── Fig. 1: Coeficientes h[n] ─────────────────
fig1, axes = plt.subplots(1, 2, figsize=(13, 5))
fig1.subplots_adjust(wspace=0.35)

axes[0].stem(range(N_TAPS), h_hp, linefmt='k-', markerfmt='ko', basefmt='k-')
axes[0].set_title(f'Coeficientes $h[n]$ — Paso Alto ({N_TAPS} taps)')
axes[0].set_xlabel('$n$ (tap)')
axes[0].set_ylabel('Amplitud')
axes[0].grid(True)

axes[1].stem(range(N_TAPS), h_bp, linefmt='k-', markerfmt='ks', basefmt='k-')
axes[1].set_title(f'Coeficientes $h[n]$ — Paso Banda ({N_TAPS} taps)')
axes[1].set_xlabel('$n$ (tap)')
axes[1].set_ylabel('Amplitud')
axes[1].grid(True)

fig1.suptitle('Fig. 1 — Coeficientes de los filtros FIR', fontsize=14, y=1.02)
fig1.tight_layout()
plt.show()

# ── Fig. 2: Respuestas en frecuencia ──────────
fig2, axes = plt.subplots(1, 2, figsize=(13, 5))
fig2.subplots_adjust(wspace=0.35)

axes[0].plot(w_hp / 1e3, H_hp_dB, **ESTILO_HP)
axes[0].axvline(FC_HP / 1e3, color='black', linewidth=1.0, linestyle='-.', label=f'$f_c$ = {FC_HP/1e3:.0f} kHz')
axes[0].axhline(-40, color='black', linewidth=0.8, linestyle=':', label='−40 dB')
axes[0].axhline(-1,  color='gray',  linewidth=0.8, linestyle=':', label='−1 dB')
axes[0].set_ylim(-90, 5)
axes[0].set_xlim(0, 500)
axes[0].set_title('Respuesta en Frecuencia — Paso Alto')
axes[0].set_xlabel('Frecuencia (kHz)')
axes[0].set_ylabel('$|H(f)|$ (dB)')
axes[0].legend(loc='lower right')
axes[0].grid(True)

axes[1].plot(w_bp / 1e3, H_bp_dB, **ESTILO_BP)
for fc in FC_BP:
    axes[1].axvline(fc / 1e3, color='black', linewidth=1.0, linestyle='-.')
axes[1].axhline(-40, color='black', linewidth=0.8, linestyle=':', label='−40 dB')
axes[1].axhline(-1,  color='gray',  linewidth=0.8, linestyle=':', label='−1 dB')
axes[1].set_ylim(-90, 5)
axes[1].set_xlim(0, 500)
axes[1].set_title('Respuesta en Frecuencia — Paso Banda')
axes[1].set_xlabel('Frecuencia (kHz)')
axes[1].set_ylabel('$|H(f)|$ (dB)')
axes[1].legend(loc='lower right')
axes[1].grid(True)

fig2.suptitle('Fig. 2 — Respuesta en frecuencia de los filtros FIR', fontsize=14, y=1.02)
fig2.tight_layout()
plt.show()

# ── Fig. 3: FFT antes/después ─────────────────
fig3, axes = plt.subplots(1, 2, figsize=(13, 5))
fig3.subplots_adjust(wspace=0.35)

axes[0].plot(freqs_mt  / 1e3, Xmt,     **ESTILO_ENT)
axes[0].plot(freqs_hpo / 1e3, Xhp_out, **ESTILO_SAL_HP)
axes[0].set_xlim(0, 450)
axes[0].set_title('FFT — Filtro Paso Alto')
axes[0].set_xlabel('Frecuencia (kHz)')
axes[0].set_ylabel('Amplitud normalizada')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(freqs_mt  / 1e3, Xmt,     **ESTILO_ENT)
axes[1].plot(freqs_bpo / 1e3, Xbp_out, **ESTILO_SAL_BP)
axes[1].set_xlim(0, 450)
axes[1].set_title('FFT — Filtro Paso Banda')
axes[1].set_xlabel('Frecuencia (kHz)')
axes[1].set_ylabel('Amplitud normalizada')
axes[1].legend()
axes[1].grid(True)

fig3.suptitle('Fig. 3 — Espectro de frecuencia antes y después del filtrado', fontsize=14, y=1.02)
=======================================================
  SEMANA 2 — Filtros FIR: Paso Alto y Paso Banda
=======================================================

[Paso Alto] fc=200 kHz | 65 taps | hamming
  @ 50 kHz  (stopband) : -60.3 dB
  @ 300 kHz (passband) : 0.0 dB
  Ripple passband      : 0.02 dB

[Paso Banda] fc=[150,250] kHz | 65 taps | hamming
  @ 50 kHz  (stopband) : -93.2 dB
  @ 200 kHz (passband) : -0.0 dB
  @ 300 kHz (stopband) : -55.6 dB
  Ripple passband      : 6.03 dB
=======================================================
fig3.tight_layout()
plt.show()
