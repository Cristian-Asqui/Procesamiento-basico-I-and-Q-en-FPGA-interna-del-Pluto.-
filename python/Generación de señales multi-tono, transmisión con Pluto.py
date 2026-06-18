"""
Proyecto: Filtro Digital FIR en Python para Señales SDR
Semana 3: Generación de señal multi-tono y transmisión con ADALM-Pluto
Universidad Nacional de Chimborazo - Escuela de Telecomunicaciones

Especificaciones:
  - Señal multi-tono : 50 + 200 + 300 kHz
  - Sample rate      : 1 MHz
  - Frecuencia central TX : 915 MHz
  - IP del Pluto     : 192.168.2.1

Dependencias:
  pip install pyadi-iio numpy scipy matplotlib
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import adi


# CONFIGURACIÓN MATPLOTLIB — Lineamientos IEEE

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


# PARÁMETROS

FS          = 1_000_000       # Sample rate: 1 MHz
FC_TX       = 915_000_000     # Frecuencia central TX: 915 MHz
TX_GAIN     = -30             # Ganancia TX en dB (negativo = menor potencia)
N           = 2**14           # Número de muestras (potencia de 2)
t           = np.arange(N) / FS

# Frecuencias de los tonos
F1 = 50e3
F2 = 200e3
F3 = 300e3


# 1. GENERACIÓN DE SEÑAL MULTI-TONO I/Q
# Señal compleja: cada tono como exponencial compleja
tono1 = np.exp(1j * 2 * np.pi * F1 * t)
tono2 = np.exp(1j * 2 * np.pi * F2 * t)
tono3 = np.exp(1j * 2 * np.pi * F3 * t)

señal_iq = tono1 + tono2 + tono3

# Normalizar a rango [-1, 1] y escalar a int16 para el Pluto
señal_norm = señal_iq / np.max(np.abs(señal_iq))
señal_tx   = (señal_norm * 2**14).astype(np.complex64)

print("=" * 55)
print("  SEMANA 3 — Transmisión multi-tono con ADALM-Pluto")
print("=" * 55)
print(f"\n  Tonos         : {F1/1e3:.0f} kHz, {F2/1e3:.0f} kHz, {F3/1e3:.0f} kHz")
print(f"  Sample rate   : {FS/1e6:.1f} MHz")
print(f"  Frec. central : {FC_TX/1e6:.0f} MHz")
print(f"  Muestras      : {N}")
print(f"  Ganancia TX   : {TX_GAIN} dB")


# 2. VERIFICACIÓN LOCAL CON FFT

X     = fft(señal_norm)
freqs = fftfreq(N, d=1/FS)
idx   = np.argsort(freqs)
freqs = freqs[idx]
X     = X[idx]
X_dB  = 20 * np.log10(np.abs(X) / N + 1e-12)

print("\n  Verificación FFT local:")
for f in [F1, F2, F3]:
    i = np.argmin(np.abs(freqs - f))
    print(f"    @ {f/1e3:.0f} kHz : {X_dB[i]:.1f} dB")


# 3. FIGURA — Señal generada y espectro

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.subplots_adjust(wspace=0.35)

# Parte real de la señal en el tiempo
axes[0].plot(t[:500] * 1e6, señal_norm[:500].real,
             color='black', linestyle='-', linewidth=1.2)
axes[0].set_title('Señal multi-tono — Parte Real (primeras 500 muestras)')
axes[0].set_xlabel('Tiempo (µs)')
axes[0].set_ylabel('Amplitud normalizada')
axes[0].grid(True)

# Espectro FFT
axes[1].plot(freqs / 1e3, X_dB,
             color='black', linestyle='-', linewidth=1.2)
axes[1].set_xlim(-500, 500)
axes[1].set_ylim(-80, 10)
axes[1].set_title('Espectro FFT — Señal multi-tono')
axes[1].set_xlabel('Frecuencia (kHz)')
axes[1].set_ylabel('Magnitud (dB)')
# Marcar los 3 tonos
for f, lbl in [(F1, '50k'), (F2, '200k'), (F3, '300k'),
               (-F1, '-50k'), (-F2, '-200k'), (-F3, '-300k')]:
    axes[1].axvline(f / 1e3, color='gray', linestyle='--', linewidth=0.8)
axes[1].grid(True)

fig.suptitle('Fig. 1 — Señal I/Q multi-tono generada para transmisión', fontsize=14, y=1.02)
fig.tight_layout()
plt.show()


# 4. CONEXIÓN Y TRANSMISIÓN CON EL PLUTO

print("\n  Conectando al ADALM-Pluto en ip:192.168.2.1 ...")

try:
    sdr = adi.Pluto("ip:192.168.2.1")

    # Configurar transmisor
    sdr.sample_rate        = FS
    sdr.tx_rf_bandwidth    = FS
    sdr.tx_lo              = FC_TX
    sdr.tx_hardwaregain_chan0 = TX_GAIN
    sdr.tx_cyclic_buffer   = True    # transmisión continua en loop

    print(f"  Pluto conectado ✅")
    print(f"  Sample rate   : {sdr.sample_rate / 1e6:.1f} MHz")
    print(f"  TX LO         : {sdr.tx_lo / 1e6:.0f} MHz")
    print(f"  TX Gain       : {sdr.tx_hardwaregain_chan0} dB")

    # Transmitir
    sdr.tx(señal_tx)
    print("\n  Transmitiendo señal multi-tono... (Ctrl+C para detener)")

    input("\n  Presiona Enter para detener la transmisión...")

except KeyboardInterrupt:
    print("\n  Transmisión detenida por el usuario.")

except Exception as e:
    print(f"\n  Error al conectar con el Pluto: {e}")

finally:
    try:
        sdr.tx_destroy_buffer()
        print("  Buffer TX liberado.")
    except:
        pass

print("\n  Semana 3 completada.")
print("=" * 55)
# 4. respuesta de la consola 
=======================================================
  SEMANA 3 — Transmisión multi-tono con ADALM-Pluto
=======================================================

  Tonos         : 50 kHz, 200 kHz, 300 kHz
  Sample rate   : 1.0 MHz
  Frec. central : 915 MHz
  Muestras      : 16384
  Ganancia TX   : -30 dB

  Verificación FFT local:
    @ 50 kHz : -10.1 dB
    @ 200 kHz : -10.1 dB
    @ 300 kHz : -10.1 dB

  Conectando al ADALM-Pluto en ip:192.168.2.1 ...
  Pluto conectado ✅
  Sample rate   : 1.0 MHz
  TX LO         : 915 MHz
  TX Gain       : -30 dB

  Transmitiendo señal multi-tono... (Ctrl+C para detener)

  Presiona Enter para detener la transmisión...
  Buffer TX liberado.

  Semana 3 completada.
