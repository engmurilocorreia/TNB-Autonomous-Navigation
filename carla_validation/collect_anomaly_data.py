import carla, random, pygame, numpy as np, csv
from datetime import datetime

# ===================================================
# CONFIGURAÇÃO INICIAL
# ===================================================
client = carla.Client('localhost', 2000)
world = client.get_world()

settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

tm = client.get_trafficmanager()
tm.set_synchronous_mode(True)

# ===================================================
# PROCESSADORES DE SENSORES (IMU + COLISÃO)
# ===================================================
class IMUProcessor:
    def __init__(self):
        self.acc = None
        self.ts = None

    def imu_callback(self, data):
        self.acc = np.array([data.accelerometer.x, data.accelerometer.y, data.accelerometer.z])
        self.ts = data.timestamp

    def get_latest(self):
        if self.ts is None: return None
        return {"timestamp": self.ts, "accelerometer": self.acc}

class CollisionProcessor:
    def __init__(self):
        self.is_collision = 0 # 0 = Normal, 1 = Anomalia (Batida)

    def collision_callback(self, event):
        self.is_collision = 1 # Registra a batida!

# ===================================================
# SPAWN VEÍCULOS (LÓGICA ORIGINAL ADAPTADA)
# ===================================================
spawn_pts = world.get_map().get_spawn_points()
all_vehicles = []
ego = None

# 1. Spawn do EGO (Sem autopilot, para você dirigir)
for pt in random.sample(spawn_pts, len(spawn_pts)):
    bp = random.choice(world.get_blueprint_library().filter('vehicle.*'))
    ego = world.try_spawn_actor(bp, pt)
    if ego:
        all_vehicles.append(ego)
        break

if not ego:
    raise RuntimeError("Não foi possível spawnar o veículo ego!")

# 2. Spawn do Tráfego (Com autopilot, para servirem de alvo)
for _ in range(50):
    pt = random.choice(spawn_pts)
    bp = random.choice(world.get_blueprint_library().filter('vehicle.*'))
    v = world.try_spawn_actor(bp, pt)
    if v:
        all_vehicles.append(v)
        v.set_autopilot(True)
        tm.ignore_lights_percentage(v, 30)
        tm.random_left_lanechange_percentage(v, 50)

# ===================================================
# SENSORES NO EGO VEHICLE
# ===================================================
# 1. IMU
imu_proc = IMUProcessor()
imu_bp = world.get_blueprint_library().find('sensor.other.imu')
imu_bp.set_attribute('sensor_tick', '0.05')
imu_sensor = world.spawn_actor(imu_bp, carla.Transform(carla.Location(z=0)), attach_to=ego)
imu_sensor.listen(lambda d: imu_proc.imu_callback(d))

# 2. COLISÃO
coll_proc = CollisionProcessor()
coll_bp = world.get_blueprint_library().find('sensor.other.collision')
coll_sensor = world.spawn_actor(coll_bp, carla.Transform(), attach_to=ego)
coll_sensor.listen(lambda e: coll_proc.collision_callback(e))

# 3. CÂMERA (Para você ver onde bate)
cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
cam_bp.set_attribute('image_size_x', '800')
cam_bp.set_attribute('image_size_y', '600')
camera = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=-5, z=3), carla.Rotation(pitch=-20)), attach_to=ego)
camera_surface = pygame.Surface((800, 600))

def cam_cb(image):
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))[:, :, :3]
    pg_img = pygame.image.frombuffer(arr.tobytes(), (image.width, image.height), 'RGB')
    camera_surface.blit(pg_img, (0, 0))
camera.listen(cam_cb)

# ===================================================
# FUNÇÕES MATEMÁTICAS TNB
# ===================================================
WINDOW_SIZE = 50
imu_buf = []

def estimate_parameters(X, n_iter=5):
    mu = np.nanmean(X[:, 1:])
    alpha = 0.0
    for _ in range(n_iter):
        num = np.nansum((X[:, 1:] - mu) * X[:, :-1])
        den = np.nansum((X[:, :-1] - mu) ** 2)
        alpha = 0.0 if den == 0 else num / den
        mu = np.nanmean(X[:, 1:] - alpha * X[:, :-1])
    residuals = X[:, 1:] - (mu + alpha * X[:, :-1])
    sigma = np.sqrt(np.nanmean(residuals ** 2))
    return mu, sigma, alpha

# ===================================================
# LOGGING CSV
# ===================================================
import os
os.makedirs("carla_validation/data", exist_ok=True)
logf = open("carla_validation/data/anomaly_dataset.csv", "w", newline='')
logw = csv.writer(logf)
logw.writerow(["timestamp", "mu", "sigma", "alpha", "is_anomaly"])

# ===================================================
# LOOP PRINCIPAL (PYGAME)
# ===================================================
pygame.init()
screen = pygame.display.set_mode((1100, 600))
pygame.display.set_caption("TNB Anomaly Collector - DIRIJINDO MANUALMENTE")
font = pygame.font.Font(None, 36)
clock = pygame.time.Clock()

print(">>> INICIANDO COLETA. USE W, A, S, D PARA DIRIGIR E BATER O CARRO! <<<")

running = True
while running:
    world.tick()
    
    # --- CONTROLE MANUAL DO CARRO ---
    keys = pygame.key.get_pressed()
    control = carla.VehicleControl()
    control.throttle = 1.0 if keys[pygame.K_w] or keys[pygame.K_UP] else 0.0
    control.brake = 1.0 if keys[pygame.K_s] or keys[pygame.K_DOWN] else 0.0
    control.steer = -0.5 if keys[pygame.K_a] or keys[pygame.K_LEFT] else (0.5 if keys[pygame.K_d] or keys[pygame.K_RIGHT] else 0.0)
    ego.apply_control(control)

    # --- PROCESSAMENTO IMU + TNB ---
    nd = imu_proc.get_latest()
    mu_est, sigma_est, alpha_est = 0.0, 0.0, 0.0
    
    if nd:
        imu_buf.append(nd)
        if len(imu_buf) > WINDOW_SIZE: imu_buf.pop(0)
            
    if len(imu_buf) == WINDOW_SIZE:
        mag = np.array([np.linalg.norm(x["accelerometer"]) for x in imu_buf])
        wd = mag.reshape(1, -1)
        mu_est, sigma_est, alpha_est = estimate_parameters(wd)
        
        # CAPTURA O STATUS DE ANOMALIA E SALVA
        anomaly_flag = coll_proc.is_collision
        logw.writerow([nd['timestamp'], f"{mu_est:.4f}", f"{sigma_est:.4f}", f"{alpha_est:.4f}", anomaly_flag])
        
        # Reseta o sensor de colisão para a próxima janela
        coll_proc.is_collision = 0 

    # --- DESENHA NA TELA ---
    screen.fill((0, 0, 0))
    screen.blit(camera_surface, (0, 0))
    
    panel = pygame.Surface((300, 600)); panel.fill((30, 30, 30))
    y = 50
    panel.blit(font.render("TNB PARAMETERS", True, (200, 200, 200)), (20, y)); y += 50
    panel.blit(font.render(f"σ: {sigma_est:.2f}", True, (255, 255, 255)), (20, y)); y += 50
    
    if coll_proc.is_collision == 1 or sigma_est > 1.0:
        panel.blit(font.render("!!! CRASH !!!", True, (255, 0, 0)), (20, y))
        
    screen.blit(panel, (800, 0))
    pygame.display.flip()

    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            running = False

    clock.tick(20)

# ===================================================
# LIMPEZA
# ===================================================
logf.close()
pygame.quit()
imu_sensor.destroy()
coll_sensor.destroy()
camera.destroy()
for v in all_vehicles: v.destroy()
print(">>> DADOS SALVOS EM 'carla_validation/data/anomaly_dataset.csv' <<<")