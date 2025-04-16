import carla
import random
import pygame
import numpy as np
import os
import shutil
from datetime import datetime

# ===================================================
# CONFIGURAÇÃO INICIAL
# ===================================================
DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "training")
VAL_DIR = os.path.join(DATA_DIR, "validation")

if os.path.exists(DATA_DIR):
    shutil.rmtree(DATA_DIR)
os.makedirs(os.path.join(TRAIN_DIR, "imu"))
os.makedirs(os.path.join(TRAIN_DIR, "labels"))
os.makedirs(os.path.join(VAL_DIR, "imu"))
os.makedirs(os.path.join(VAL_DIR, "labels"))

client = carla.Client('localhost', 2000)
world = client.get_world()

# ===================================================
# CONFIGURAÇÃO DO MODO SÍNCRONO
# ===================================================
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
settings.no_rendering_mode = True  # Se preferir visualizar o ambiente, remova ou comente esta linha.
world.apply_settings(settings)

traffic_manager = client.get_trafficmanager()
traffic_manager.set_synchronous_mode(True)
traffic_manager.set_global_distance_to_leading_vehicle(1.0)
traffic_manager.global_percentage_speed_difference(-30)

# ===================================================
# CLASSE: IMUProcessor (atualizado)
# ===================================================
class IMUProcessor:
    def __init__(self):
        self.accelerometer = None
        self.gyroscope = None
        self.compass = None
        self.timestamp = None

    def imu_callback(self, data):
        self.accelerometer = np.array([data.accelerometer.x, data.accelerometer.y, data.accelerometer.z])
        self.gyroscope = np.array([data.gyroscope.x, data.gyroscope.y, data.gyroscope.z])
        self.compass = data.compass
        self.timestamp = data.timestamp
        print(f"Time: {self.timestamp:.2f} | Acc: {self.accelerometer} | Gyro: {self.gyroscope} | Compass: {self.compass:.2f}")

    def get_latest(self):
        if self.timestamp is not None:
            return {
                "timestamp": self.timestamp,
                "accelerometer": self.accelerometer,
                "gyroscope": self.gyroscope,
                "compass": self.compass
            }
        else:
            return None

# ===================================================
# CONFIGURAÇÃO DOS VEÍCULOS
# ===================================================
spawn_points = world.get_map().get_spawn_points()
all_vehicles = []

ego_vehicle = None
for spawn_point in random.sample(spawn_points, len(spawn_points)):
    bp = random.choice(world.get_blueprint_library().filter('vehicle.*'))
    ego_vehicle = world.try_spawn_actor(bp, spawn_point)
    if ego_vehicle is not None:
        all_vehicles.append(ego_vehicle)
        break

if ego_vehicle is None:
    raise RuntimeError("Não foi possível spawnar o veículo ego!")

for _ in range(90):
    spawn_point = random.choice(spawn_points)
    bp = random.choice(world.get_blueprint_library().filter('vehicle.*'))
    vehicle = world.try_spawn_actor(bp, spawn_point)
    if vehicle is not None:
        all_vehicles.append(vehicle)

for vehicle in all_vehicles:
    vehicle.set_autopilot(True)
    traffic_manager.ignore_lights_percentage(vehicle, 30)
    traffic_manager.random_left_lanechange_percentage(vehicle, 50)

# ===================================================
# CONFIGURAÇÃO DOS SENSORES
# ===================================================
imu_processor = IMUProcessor()
imu_bp = world.get_blueprint_library().find('sensor.other.imu')
imu_bp.set_attribute('sensor_tick', '0.05')  # 20 Hz, por exemplo
imu_transform = carla.Transform(carla.Location(x=0, y=0, z=0))
imu_sensor = world.spawn_actor(imu_bp, imu_transform, attach_to=ego_vehicle)
imu_sensor.listen(lambda data: imu_processor.imu_callback(data))

# ===================================================
# CONFIGURAÇÃO DO BUFFER PARA PROCESSAMENTO EM TEMPO REAL
# ===================================================
WINDOW_SIZE = 50  
imu_buffer = []  # Cada item é um dicionário com os dados do IMU

def update_buffer(new_data, buffer, window_size):
    buffer.append(new_data)
    if len(buffer) > window_size:
        buffer.pop(0)
    return buffer

def compute_acc_magnitude(buffer):
    acc_mag = []
    timestamps = []
    for data in buffer:
        m = np.linalg.norm(data["accelerometer"])
        acc_mag.append(m)
        timestamps.append(data["timestamp"])
    return np.array(timestamps), np.array(acc_mag)

def estimate_parameters(X, n_iter=5):
    mu = np.mean(X[:, 1:])
    alpha = 0.0
    for _ in range(n_iter):
        alpha = np.sum((X[:, 1:] - mu) * X[:, :-1]) / np.sum((X[:, :-1] - mu) ** 2)
        mu = np.mean(X[:, 1:] - alpha * X[:, :-1])
    residuals = X[:, 1:] - (mu + alpha * X[:, :-1])
    sigma = np.sqrt(np.mean(residuals**2))
    return mu, sigma, alpha

# ===================================================
# CONFIGURAÇÃO DA INTERFACE GRÁFICA COM PYGAME
# ===================================================
pygame.init()
screen = pygame.display.set_mode((800, 600))
font = pygame.font.Font(None, 24)
clock = pygame.time.Clock()

# ===================================================
# LOOP PRINCIPAL DO SCRIPT EM TEMPO REAL
# ===================================================
running = True
while running:
    world.tick()
    
    # Obtém o dado IMU mais recente a partir do IMUProcessor
    new_imu_data = imu_processor.get_latest()  # Método agora implementado
    if new_imu_data is not None:
        imu_buffer = update_buffer(new_imu_data, imu_buffer, WINDOW_SIZE)
    
    # Quando o buffer estiver completo, processa a janela
    if len(imu_buffer) == WINDOW_SIZE:
        timestamps, acc_mag = compute_acc_magnitude(imu_buffer)
        window_data = acc_mag.reshape(1, -1)
        mu_est, sigma_est, alpha_est = estimate_parameters(window_data)
    
    # Atualiza a interface: exibe os parâmetros estimados, se disponíveis
    screen.fill((0, 0, 0))
    if len(imu_buffer) == WINDOW_SIZE:
        info_text = f"μ = {mu_est:.2f}, σ = {sigma_est:.2f}, α = {alpha_est:.2f}"
    else:
        info_text = "Coletando dados..."
    text_surface = font.render(info_text, True, (255, 255, 255))
    screen.blit(text_surface, (20, 20))
    
    pygame.display.flip()
    clock.tick(20)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()

# Encerramento: destruição dos atores
imu_sensor.destroy()
for vehicle in all_vehicles:
    vehicle.destroy()
