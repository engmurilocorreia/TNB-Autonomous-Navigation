import os
import numpy as np
import matplotlib.pyplot as plt

# Função de estimação de parâmetros (como definida anteriormente)
def estimate_parameters(X, n_iter=5):
    """Estimativa iterativa de mu, sigma e alpha para uma sequência 1D.
       Aqui X deve ser de shape (1, T) para representar uma única sequência."""
    mu = np.mean(X[:, 1:])  # Estimativa inicial de mu
    alpha = 0.0  # Inicialização
    
    # Iteração para estimar mu e alpha
    for _ in range(n_iter):
        alpha = np.sum((X[:, 1:] - mu) * X[:, :-1]) / np.sum((X[:, :-1] - mu) ** 2)
        mu = np.mean(X[:, 1:] - alpha * X[:, :-1])
    
    # Estimativa de sigma usando o RMS dos resíduos
    residuals = X[:, 1:] - (mu + alpha * X[:, :-1])
    sigma = np.sqrt(np.mean(residuals**2))
    
    return mu, sigma, alpha

# --------------------------------------------------
# Etapa 1: Carregar os dados já pré-processados
# --------------------------------------------------
# Supondo que o script offline_preprocessing_imu.py foi executado e os dados foram reunidos
# Aqui vamos ler todos os arquivos .npz e reconstruir as séries temporais do IMU

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data", "training", "imu")
files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npz")])

print(f"Encontrados {len(files)} arquivos .npz")

timestamps = []
acc_x, acc_y, acc_z = [], [], []
for file in files:
    file_path = os.path.join(data_dir, file)
    with np.load(file_path) as data:
        if all(k in data for k in ['timestamp', 'accelerometer']):
            t = data['timestamp'].item()
            accel = data['accelerometer']
            timestamps.append(t)
            acc_x.append(accel[0])
            acc_y.append(accel[1])
            acc_z.append(accel[2])
        else:
            print(f"Aviso: {file} não possui as chaves necessárias.")

# Converter para arrays e ordenar pelo timestamp
timestamps = np.array(timestamps)
ord_idx = np.argsort(timestamps)
timestamps = timestamps[ord_idx]
acc_x = np.array(acc_x)[ord_idx]
acc_y = np.array(acc_y)[ord_idx]
acc_z = np.array(acc_z)[ord_idx]

# Normaliza os timestamps para iniciar em zero
timestamps = timestamps - timestamps[0]

# --------------------------------------------------
# Etapa 2: Pré-processamento - Construir a magnitude da aceleração
# --------------------------------------------------
# Para simplificar, vamos usar a magnitude da aceleração:
acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

# --------------------------------------------------
# Etapa 3: Dividir a série em janelas (sliding window)
# --------------------------------------------------
window_size = 50     # Pode ser ajustado conforme necessário (por exemplo, 50 amostras = 50*fixed_delta_seconds)
step_size   = 10     # Deslocamento entre as janelas

num_windows = (len(acc_mag) - window_size) // step_size + 1

est_mu = []       # Para armazenar os valores estimados de mu
est_sigma = []    # Para sigma
est_alpha = []    # Para alpha
time_centers = [] # Para o timestamp central de cada janela

for i in range(num_windows):
    start = i * step_size
    end = start + window_size
    window_data = acc_mag[start:end]
    
    # Reshape para (1, T) para compatibilidade com a função de estimação
    window_data = window_data.reshape(1, -1)
    
    mu, sigma, alpha = estimate_parameters(window_data)
    est_mu.append(mu)
    est_sigma.append(sigma)
    est_alpha.append(alpha)
    
    # Timestamp central da janela
    time_centers.append(timestamps[start + window_size // 2])

est_mu = np.array(est_mu)
est_sigma = np.array(est_sigma)
est_alpha = np.array(est_alpha)
time_centers = np.array(time_centers)

# --------------------------------------------------
# Etapa 4: Visualização dos parâmetros estimados ao longo do tempo
# --------------------------------------------------
plt.figure(figsize=(14, 10))

plt.subplot(3,1,1)
plt.plot(time_centers, est_mu, 'b.-')
plt.title('Estimativas de μ ao Longo do Tempo')
plt.xlabel('Tempo [s]')
plt.ylabel('μ')
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(time_centers, est_sigma, 'r.-')
plt.title('Estimativas de σ ao Longo do Tempo')
plt.xlabel('Tempo [s]')
plt.ylabel('σ')
plt.grid(True)

plt.subplot(3,1,3)
plt.plot(time_centers, est_alpha, 'g.-')
plt.title('Estimativas de α ao Longo do Tempo')
plt.xlabel('Tempo [s]')
plt.ylabel('α')
plt.grid(True)

plt.tight_layout()
plt.show()
