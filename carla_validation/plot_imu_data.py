import os
import numpy as np
import matplotlib.pyplot as plt

# Caminho relativo ao local do script
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data", "training", "imu")

# Inicializa listas para acumular os dados
acc_x, acc_y, acc_z = [], [], []
gyro_x, gyro_y, gyro_z = [], [], []
timestamps = []

# Lê todos os arquivos .npz da pasta, ordenados
files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npz")])

print(f"Encontrados {len(files)} arquivos .npz")

for file in files:
    data = np.load(os.path.join(data_dir, file))

    if all(k in data for k in ['accelerometer', 'gyroscope', 'timestamp']):
        accel = data['accelerometer']
        gyro = data['gyroscope']
        t = data['timestamp']

        acc_x.append(accel[0])
        acc_y.append(accel[1])
        acc_z.append(accel[2])

        gyro_x.append(gyro[0])
        gyro_y.append(gyro[1])
        gyro_z.append(gyro[2])

        timestamps.append(t)
    else:
        print(f"Aviso: {file} não possui todas as chaves esperadas.")

# Converte timestamps para numpy array
timestamps = np.array(timestamps)
timestamps -= timestamps[0]  # Opcional: começa em t = 0

# Plota aceleração
plt.figure(figsize=(14, 6))
plt.subplot(2, 1, 1)
plt.plot(timestamps, acc_x, label="Acc X")
plt.plot(timestamps, acc_y, label="Acc Y")
plt.plot(timestamps, acc_z, label="Acc Z")
plt.title("Aceleração Linear")
plt.xlabel("Tempo [s]")
plt.ylabel("m/s²")
plt.legend()
plt.grid(True)

# Plota giroscópio
plt.subplot(2, 1, 2)
plt.plot(timestamps, gyro_x, label="Gyro X")
plt.plot(timestamps, gyro_y, label="Gyro Y")
plt.plot(timestamps, gyro_z, label="Gyro Z")
plt.title("Velocidade Angular")
plt.xlabel("Tempo [s]")
plt.ylabel("rad/s")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
