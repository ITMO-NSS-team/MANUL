import numpy as np
import matplotlib.pyplot as plt

# Загрузка данных
H = np.load('sionna_sample.npy')  # [64, 4, 256, 20, 4]
print(f"Shape: {H.shape}")
print(f"Data type: {H.dtype}")

# 1. Схлопываем последние две размерности
H_flat = H.reshape(64, 4, 256, -1)  # [64, 4, 256, 80]
print(f"После схлопывания: {H_flat.shape}")
# 20 пользователей × 4 сценария

amplitude = np.abs(H_flat)        # [64, 4, 256, 80] - сила сигнала
phase = np.angle(H_flat)          # [64, 4, 256, 80] - фазовые сдвиги

# Объединяем в один тензор
H_processed = np.stack([amplitude, phase], axis=-1)  # [64, 4, 256, 80, 2]
print(f"Амплитуда+Фаза: {H_processed.shape}")

user_idx = 0
fig, axes = plt.subplots(8, 8, figsize=(20, 21))
axes = axes.flatten()
for s in range(64):  # Для каждой станции
    ax = axes[s]
    # Рисуем амплитуды для всех 4 антенн UE
    for i in range(4):
        ax.plot(amplitude[s, i, :, user_idx], alpha=0.7, linewidth=1, label=f'UE ant {i}')
    ax.set_ylim(0, 2.2)
    ax.set_title(f'BS Ant {s}')
    ax.set_ylabel('Amplitude')
    ax.set_xlabel('Freq')
plt.suptitle('Amplitude for all 64 BS Antennas (User 0)', fontsize=16)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(amplitude.flatten(), bins=50, alpha=0.7)
plt.title('Распределение амплитуд')
plt.xlabel('Амплитуда')
plt.ylabel('Частота')
plt.subplot(1, 2, 2)
plt.hist(phase.flatten(), bins=50, alpha=0.7)
plt.title('Распределение фаз')
plt.xlabel('Фаза (радианы)')
plt.ylabel('Частота')
plt.tight_layout()
plt.show()