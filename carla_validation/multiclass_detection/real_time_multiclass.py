import carla
import random
import pygame
import numpy as np
import os
import shutil
import json
from datetime import datetime


# Carrega parâmetros
with open('class_params.json', 'r') as f:
    class_params = json.load(f)

# ===================================================
# CONFIGURAÇÃO INICIAL
# ===================================================
DATA_DIR = "data"
for d in ["training/imu","training/labels","validation/imu","validation/labels"]:
    os.makedirs(os.path.join(DATA_DIR, d), exist_ok=True)

client = carla.Client('localhost', 2000)
world = client.get_world()

# ===================================================
# CONFIGURAÇÃO DO MODO SÍNCRONO
# ===================================================
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

tm = client.get_trafficmanager()
tm.set_synchronous_mode(True)
tm.set_global_distance_to_leading_vehicle(1.0)
tm.global_percentage_speed_difference(-30)

# ===================================================
# CLASSE IMUProcessor
# ===================================================
class IMUProcessor:
    def __init__(self):
        self.acc = None
        self.gyro = None
        self.ts = None

    def imu_callback(self, data):
        self.acc = np.array([data.accelerometer.x,
                             data.accelerometer.y,
                             data.accelerometer.z])
        self.gyro = np.array([data.gyroscope.x,
                              data.gyroscope.y,
                              data.gyroscope.z])
        self.ts = data.timestamp

    def get_latest(self):
        if self.ts is None:
            return None
        return {"timestamp": self.ts,
                "accelerometer": self.acc,
                "gyroscope": self.gyro}

# ===================================================
# SPAWN VEÍCULO E NPCs
# ===================================================
spawn_pts = world.get_map().get_spawn_points()
all_vehicles = []
# Spawn ego
ego_bp = random.choice(world.get_blueprint_library().filter('vehicle.*'))
ego = world.try_spawn_actor(ego_bp, spawn_pts[0])
all_vehicles.append(ego)
# Spawn poucos NPCs
for _ in range(5):
    pt = random.choice(spawn_pts)
    bp = random.choice(world.get_blueprint_library().filter('vehicle.*'))
    v = world.try_spawn_actor(bp, pt)
    if v:
        all_vehicles.append(v)
for v in all_vehicles:
    v.set_autopilot(True)
    tm.ignore_lights_percentage(v, 30)
    tm.random_left_lanechange_percentage(v, 50)

# ===================================================
# SENSORES IMU + CÂMERA
# ===================================================
imu_proc = IMUProcessor()
imu_bp = world.get_blueprint_library().find('sensor.other.imu')
imu_bp.set_attribute('sensor_tick', '0.05')
imu_sensor = world.spawn_actor(imu_bp, carla.Transform(carla.Location(z=0)), attach_to=ego)
imu_sensor.listen(lambda d: imu_proc.imu_callback(d))

cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
cam_bp.set_attribute('image_size_x', '800')
cam_bp.set_attribute('image_size_y', '600')
camera = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=-5, z=3), carla.Rotation(pitch=-20)), attach_to=ego)
camera_surface = pygame.Surface((800,600))
def cam_cb(image):
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:,:,:3]
    pg_img = pygame.image.frombuffer(arr.tobytes(), (image.width, image.height), 'RGB')
    camera_surface.blit(pg_img, (0,0))
camera.listen(cam_cb)

# ===================================================
# FUNÇÕES DE PROCESSAMENTO
# ===================================================
WINDOW_SIZE = 50
imu_buf = []

def update_buffer(d, b, ws):
    b.append(d)
    if len(b) > ws:
        b.pop(0)
    return b

def compute_acc_mag(buf):
    mag = [np.linalg.norm(x['accelerometer']) for x in buf]
    return np.array(mag)

# Estima parâmetros AR(1) por janela
def estimate_parameters(X, n_iter=5):
    mu = np.nanmean(X[1:])
    alpha = 0.0
    for _ in range(n_iter):
        num = np.nansum((X[1:]-mu)*X[:-1])
        den = np.nansum((X[:-1]-mu)**2)
        alpha = 0.0 if den==0 else num/den
        mu = np.nanmean(X[1:] - alpha*X[:-1])
    res = X[1:] - (mu + alpha*X[:-1])
    sigma = np.sqrt(np.nanmean(res**2))
    return mu, sigma, alpha

# Log-verossimilhança TNB para uma janela
def log_likelihood(X, mu, sigma, alpha):
    ll = -0.5*len(X[1:])*np.log(2*np.pi*sigma**2)
    ll += -np.nansum((X[1:]-mu-alpha*X[:-1])**2)/(2*sigma**2)
    return ll

# Classifica uma janela em multiclasses
def classify_window(X):
    scores = {}
    for cls, p in class_params.items():
        ll = log_likelihood(X, p['mu'], p['sigma'], p['alpha']) + np.log(p['prior'])
        scores[cls] = ll
    return max(scores, key=scores.get)

# ===================================================
# SETUP PYGAME
# ===================================================
pygame.init()
screen = pygame.display.set_mode((1100,600))
font = pygame.font.Font(None,24)
clock = pygame.time.Clock()

# ===================================================
# LOOP PRINCIPAL
# ===================================================
predicted = None
running = True
while running:
    world.tick()
    data = imu_proc.get_latest()
    if data:
        imu_buf = update_buffer(data, imu_buf, WINDOW_SIZE)
    if len(imu_buf) == WINDOW_SIZE:
        X = compute_acc_mag(imu_buf)
        mu_est, sigma_est, alpha_est = estimate_parameters(X)
        predicted = classify_window(X)

    # renderiza
    screen.fill((0,0,0))
    screen.blit(camera_surface, (0,0))
    panel = pygame.Surface((300,600)); panel.fill((30,30,30))
    y=10
    panel.blit(font.render(f"Predição: {predicted}" if predicted else "Predição: --", True,(255,255,255)),(10,y)); y+=30
    panel.blit(font.render(f"μ: {mu_est:.2f}" if len(imu_buf)==WINDOW_SIZE else "μ: --", True,(255,255,255)),(10,y)); y+=30
    panel.blit(font.render(f"σ: {sigma_est:.2f}" if len(imu_buf)==WINDOW_SIZE else "σ: --", True,(255,255,255)),(10,y)); y+=30
    panel.blit(font.render(f"α: {alpha_est:.2f}" if len(imu_buf)==WINDOW_SIZE else "α: --", True,(255,255,255)),(10,y));
    screen.blit(panel,(800,0))
    pygame.display.flip()
    for ev in pygame.event.get():
        if ev.type==pygame.QUIT:
            running=False
    clock.tick(20)

# cleanup
pygame.quit()
imu_sensor.destroy()
camera.destroy()
for v in all_vehicles: v.destroy()
