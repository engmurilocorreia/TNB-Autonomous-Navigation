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
settings.no_rendering_mode = True
world.apply_settings(settings)

traffic_manager = client.get_trafficmanager()
traffic_manager.set_synchronous_mode(True)
traffic_manager.set_global_distance_to_leading_vehicle(1.0)
traffic_manager.global_percentage_speed_difference(-30)

# ===================================================
# CLASSES E FUNÇÕES
# ===================================================

# Classe para processar os dados do IMU
class IMUProcessor:
    def __init__(self):
        self.accelerometer = None
        self.gyroscope = None
        self.compass = None
        self.timestamp = None

    # Callback para sensor IMU do CARLA
    def imu_callback(self, data):
        # O objeto data possui os seguintes atributos:
        # - data.accelerometer: carla.Vector3D (x, y, z)
        # - data.gyroscope: carla.Vector3D (x, y, z)
        # - data.compass: float (rotação em graus ou radianos, dependendo da versão)
        # - data.timestamp: tempo da medição
        self.accelerometer = np.array([data.accelerometer.x, data.accelerometer.y, data.accelerometer.z])
        self.gyroscope = np.array([data.gyroscope.x, data.gyroscope.y, data.gyroscope.z])
        self.compass = data.compass
        self.timestamp = data.timestamp
        # Por simplicidade, vamos apenas imprimir os dados
        print(f"Time: {self.timestamp:.2f} | Acc: {self.accelerometer} | Gyro: {self.gyroscope} | Compass: {self.compass:.2f}")

# Função para salvar os dados IMU + controles em formato .npz comprimido
def save_imu_data(imu_data, control, frame_id, base_dir, sensor_type="imu"):
    """
    imu_data: dict com chaves 'timestamp', 'accelerometer', 'gyroscope', 'compass'
    control:  carla.VehicleControl (throttle, steer, brake, etc.)
    frame_id: int, número do frame
    base_dir: pasta raiz onde vão os subdiretórios
    """
    save_path = os.path.join(base_dir, sensor_type, f"frame_{frame_id:06d}.npz")
    np.savez_compressed(
        save_path,
        timestamp=imu_data['timestamp'],
        accelerometer=imu_data['accelerometer'],
        gyroscope=imu_data['gyroscope'],
        compass=imu_data['compass'],
        # novos campos de controle
        throttle=control.throttle,
        steer=control.steer,
        brake=control.brake,
        frame_id=frame_id
    )

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

for _ in range(10): # Número de veículos spawnados
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

# Configurando o sensor IMU
imu_bp = world.get_blueprint_library().find('sensor.other.imu')
# Atributos opcionais podem ser configurados, por exemplo, a taxa de atualização:
imu_bp.set_attribute('sensor_tick', '0.05')  # 20 Hz, por exemplo

# Posicionando o sensor no veículo (ajuste a posição conforme necessário)
imu_transform = carla.Transform(carla.Location(x=0, y=0, z=0))
imu_sensor = world.spawn_actor(imu_bp, imu_transform, attach_to=ego_vehicle)
imu_sensor.listen(lambda data: imu_processor.imu_callback(data))

# ===================================================
# LOOP PRINCIPAL
# ===================================================
pygame.init()
gameDisplay = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()

frame_count = 0
SAVE_INTERVAL = 10

running = True
while running:
    world.tick()
    frame_count += 1

    gameDisplay.fill((0, 0, 0))

    # Exibição simples dos dados do IMU na tela
    font = pygame.font.Font(None, 24)
    if imu_processor.timestamp is not None:
        imu_text = f"Time: {imu_processor.timestamp:.2f} | Acc: {imu_processor.accelerometer} | Gyro: {imu_processor.gyroscope}"
    else:
        imu_text = "Aguardando dados IMU..."
    text_surface = font.render(imu_text, True, (255, 255, 255))
    gameDisplay.blit(text_surface, (20, 20))

    # A cada SAVE_INTERVAL quadros, salvamos IMU + controle
    if frame_count % SAVE_INTERVAL == 0 and imu_processor.timestamp is not None:
        # Dados IMU
        imu_data = {
            'timestamp': imu_processor.timestamp,
            'accelerometer': imu_processor.accelerometer,
            'gyroscope': imu_processor.gyroscope,
            'compass': imu_processor.compass
        }
        # Comando de controle do ego
        control = ego_vehicle.get_control()
        # Escolhe pasta de treino/validação
        base_path = TRAIN_DIR if np.random.rand() < 0.8 else VAL_DIR
        # Salva tudo em um .npz
        save_imu_data(imu_data, control, frame_count, base_path, sensor_type="imu")
        print(f"Frame {frame_count:06d} salvo com controle: throttle={control.throttle:.2f}, steer={control.steer:.2f}, brake={control.brake:.2f}")

    pygame.display.flip()
    clock.tick(20)  # Limita a 20 FPS

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()

# ===================================================
# ENCERRAMENTO
# ===================================================
imu_sensor.destroy()
for vehicle in all_vehicles:
    vehicle.destroy()
