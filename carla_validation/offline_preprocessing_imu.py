import os
import numpy as np
import matplotlib.pyplot as plt

# Define o caminho da pasta de dados (utilizando caminho relativo robusto)
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data", "training", "imu")

# Lista todos os arquivos .npz na pasta
files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npz")])
print(f"Encontrados {len(files)} arquivos .npz")

# Inicializa listas para os dados a serem processados
timestamps = []
acc_x, acc_y, acc_z = [], [], []
gyro_x, gyro_y, gyro_z = [], [], []
compass_vals = []  # Se desejar usar a bússola

# Carrega cada arquivo e extrai os dados
for file in files:
    file_path = os.path.join(data_dir, file)
    with np.load(file_path) as data:
        # Verifica se o arquivo possui as chaves necessárias
        if all(k in data for k in ['timestamp', 'accelerometer', 'gyroscope', 'compass']):
            # Lê os dados do IMU
            t = data['timestamp'].item()  # converte o valor escalar
            accel = data['accelerometer']
            gyro = data['gyroscope']
            comp = data['compass'].item()  # valor escalar
            
            timestamps.append(t)
            acc_x.append(accel[0])
            acc_y.append(accel[1])
            acc_z.append(accel[2])
            
            gyro_x.append(gyro[0])
            gyro_y.append(gyro[1])
            gyro_z.append(gyro[2])
            
            compass_vals.append(comp)
        else:
            print(f"Aviso: {file} não possui todas as chaves esperadas.")

# Ordena os dados pelo timestamp para garantir sequência temporal
timestamps = np.array(timestamps)
ord_idx = np.argsort(timestamps)
timestamps = timestamps[ord_idx]
acc_x = np.array(acc_x)[ord_idx]
acc_y = np.array(acc_y)[ord_idx]
acc_z = np.array(acc_z)[ord_idx]
gyro_x = np.array(gyro_x)[ord_idx]
gyro_y = np.array(gyro_y)[ord_idx]
gyro_z = np.array(gyro_z)[ord_idx]
compass_vals = np.array(compass_vals)[ord_idx]

# Opcional: normaliza os timestamps para iniciar em 0
timestamps = timestamps - timestamps[0]

# Plota as séries temporais do IMU
plt.figure(figsize=(14, 8))

# Aceleração
plt.subplot(3, 1, 1)
plt.plot(timestamps, acc_x, label="Acc X")
plt.plot(timestamps, acc_y, label="Acc Y")
plt.plot(timestamps, acc_z, label="Acc Z")
plt.title("Linear Acceleration (m/s²)")
plt.xlabel("Time [s]")
plt.ylabel("Acceleration")
plt.legend()
plt.grid(True)

# Giroscópio
plt.subplot(3, 1, 2)
plt.plot(timestamps, gyro_x, label="Gyro X")
plt.plot(timestamps, gyro_y, label="Gyro Y")
plt.plot(timestamps, gyro_z, label="Gyro Z")
plt.title("Angular Velocity (rad/s)")
plt.xlabel("Time [s]")
plt.ylabel("Angular Velocity")
plt.legend()
plt.grid(True)

# Compass (se aplicável)
plt.subplot(3, 1, 3)
plt.plot(timestamps, compass_vals, label="Compass", color="purple")
plt.title("Compass")
plt.xlabel("Time [s]")
plt.ylabel("Orientation")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
