import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


# ESPECIFICACIONES

fs = 2.4e6          # Frecuencia de muestreo (Hz)
fc = 100e3          # Frecuencia de corte (Hz)
transition = 50e3  # Ancho de transición (Hz)
num_taps = 64       # 64 taps
window = 'hamming'


# DISEÑO DEL FILTRO FIR

h = signal.firwin(
    num_taps,
    cutoff=fc,
    window=window,
    fs=fs,
    pass_zero='lowpass'
)


# RESPUESTA EN FRECUENCIA

f, H = signal.freqz(h, worN=8192, fs=fs)

mag_db = 20 * np.log10(np.maximum(np.abs(H), 1e-10))


# MÉTRICAS


# Banda pasante: 0 - 100 kHz
passband = mag_db[f <= fc]

# Banda de rechazo: >150 kHz
stopband = mag_db[f >= (fc + transition)]

ripple = np.max(passband) - np.min(passband)
attenuation = -np.max(stopband)

print("\nRESULTADOS")
print("="*40)
print(f"Frecuencia de corte       : {fc/1000:.0f} kHz")
print(f"Ancho de transición       : {transition/1000:.0f} kHz")
print(f"Número de taps            : {num_taps}")
print(f"Ventana                   : Hamming")
print(f"Ripple banda pasante      : {ripple:.2f} dB")
print(f"Atenuación banda rechazo  : {attenuation:.2f} dB")
print("="*40)

if ripple < 1:
    print("✓ Ripple < 1 dB : CUMPLE")
else:
    print("✗ Ripple < 1 dB : NO CUMPLE")

if attenuation > 40:
    print("✓ Atenuación > 40 dB : CUMPLE")
else:
    print("✗ Atenuación > 40 dB : NO CUMPLE")


# GRÁFICA

plt.figure(figsize=(10,6))

plt.plot(f/1000, mag_db, linewidth=2)

plt.axvline(fc/1000,
            linestyle='--',
            label='fc = 100 kHz')

plt.axvline((fc+transition)/1000,
            linestyle=':',
            label='Fin transición = 150 kHz')

plt.title('Filtro FIR Paso Bajo (64 taps, Ventana Hamming)')
plt.xlabel('Frecuencia (kHz)')
plt.ylabel('Magnitud (dB)')
plt.grid(True)
plt.legend()

plt.xlim(0, 500)
plt.ylim(-100, 5)

plt.tight_layout()

plt.savefig("respuesta_filtro_FIR.png", dpi=300)

plt.show()

RESULTADOS
========================================
Frecuencia de corte       : 100 kHz
Ancho de transición       : 50 kHz
Número de taps            : 64
Ventana                   : Hamming
Ripple banda pasante      : 6.01 dB
Atenuación banda rechazo  : 31.66 dB
========================================
✗ Ripple < 1 dB : NO CUMPLE
✗ Atenuación > 40 dB : NO CUMPLE
