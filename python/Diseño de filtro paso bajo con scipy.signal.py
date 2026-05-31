import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Parámetros del filtro
orden = 4
fc = 1000      # Frecuencia de corte (Hz)
fs = 10000     # Frecuencia de muestreo (Hz)

# Diseño del filtro Butterworth paso bajo
b, a = signal.butter(orden, fc/(fs/2), btype='low')

# Respuesta en frecuencia
w, h = signal.freqz(b, a, worN=2048)

# Convertir frecuencia angular a Hz
f = w * fs / (2 * np.pi)

# Gráfica
plt.figure(figsize=(8,5))
plt.plot(f, 20*np.log10(np.abs(h)))
plt.title('Respuesta en Frecuencia - Filtro Paso Bajo Butterworth')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Magnitud (dB)')
plt.grid(True)
plt.axvline(fc, linestyle='--')
plt.show()
