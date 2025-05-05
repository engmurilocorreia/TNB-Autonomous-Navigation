import carla, random, pygame, numpy as np, os, csv, shutil
from datetime import datetime
import matplotlib.pyplot as plt

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
# settings.no_rendering_mode = True  # comente para ver 3D
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
        self.compass = None
        self.ts = None

    def imu_callback(self, data):
        self.acc = np.array([data.accelerometer.x,
                             data.accelerometer.y,
                             data.accelerometer.z])
        self.gyro = np.array([data.gyroscope.x,
                              data.gyroscope.y,
                              data.gyroscope.z])
        self.compass = data.compass
        self.ts = data.timestamp

    def get_latest(self):
        if self.ts is None:
            return None
        return {
            "timestamp": self.ts,
            "accelerometer": self.acc,
            "gyroscope": self.gyro,
            "compass": self.compass
        }

# ===================================================
# SPAWN VEÍCULOS
# ===================================================
spawn_pts = world.get_map().get_spawn_points()
all_vehicles = []
ego = None
for pt in random.sample(spawn_pts, len(spawn_pts)):
    bp = random.choice(world.get_blueprint_library().filter('vehicle.*'))
    ego = world.try_spawn_actor(bp, pt)
    if ego:
        all_vehicles.append(ego)
        break
if not ego:
    raise RuntimeError("Não foi possível spawnar o veículo ego!")
for _ in range(50): # Número de veículos spawnados
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
imu_sensor = world.spawn_actor(
    imu_bp,
    carla.Transform(carla.Location(z=0)),
    attach_to=ego)
imu_sensor.listen(lambda d: imu_proc.imu_callback(d))

cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
cam_bp.set_attribute('image_size_x', '800')
cam_bp.set_attribute('image_size_y', '600')
camera = world.spawn_actor(
    cam_bp,
    carla.Transform(carla.Location(x=-5, z=3), carla.Rotation(pitch=-20)),
    attach_to=ego)
camera_surface = pygame.Surface((800, 600))
def cam_cb(image):
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))[:, :, :3]
    pg_img = pygame.image.frombuffer(arr.tobytes(),
                                     (image.width, image.height),
                                     'RGB')
    camera_surface.blit(pg_img, (0, 0))

camera.listen(cam_cb)

# ===================================================
# BUFFER + PROCESSAMENTO TNB
# ===================================================
WINDOW_SIZE = 50
imu_buf = []
sigma_history = []
SIGMA_ALERT = 1.0
HIST_BINS = 10

def update_buffer(d, b, ws):
    b.append(d)
    if len(b) > ws:
        b.pop(0)
    return b

def compute_acc_mag(buf):
    ts = []
    mag = []
    for x in buf:
        ts.append(x["timestamp"])
        mag.append(np.linalg.norm(x["accelerometer"]))
    return np.array(ts), np.array(mag)

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

# Logging CSV
logf = open("realtime_log.csv", "w", newline='')
logw = csv.writer(logf)
logw.writerow(["timestamp", "mu", "sigma", "alpha"])

# ===================================================
# SETUP PYGAME + MATPLOTLIB
# ===================================================
pygame.init()
screen = pygame.display.set_mode((1100, 600))
font = pygame.font.Font(None, 24)
clock = pygame.time.Clock()

plt.ion()
fig, (ax_gauge, ax_hist) = plt.subplots(2, 1, figsize=(4, 6))
ax_gauge.set_xlim(0, SIGMA_ALERT * 1.2)
ax_gauge.set_ylim(0, 1)
ax_gauge.set_yticks([])
ax_gauge.set_title('Gauge σ')
ax_hist.set_title('Histograma σ')
ax_hist.set_xlim(0, SIGMA_ALERT)

mu_est = sigma_est = alpha_est = None

# ===================================================
# LOOP PRINCIPAL
# ===================================================
running = True
while running:
    world.tick()
    nd = imu_proc.get_latest()
    if nd:
        imu_buf = update_buffer(nd, imu_buf, WINDOW_SIZE)
    if len(imu_buf) == WINDOW_SIZE:
        ts, mag = compute_acc_mag(imu_buf)
        wd = mag.reshape(1, -1)
        mu_est, sigma_est, alpha_est = estimate_parameters(wd)
        sigma_history.append(sigma_est)
        logw.writerow([nd['timestamp'], f"{mu_est:.4f}", f"{sigma_est:.4f}", f"{alpha_est:.4f}"])
        if len(sigma_history) > 1000:
            sigma_history.pop(0)

        # atualiza Matplotlib
        ax_gauge.clear()
        ax_gauge.barh([0], [sigma_est], color='green')
        ax_gauge.set_xlim(0, SIGMA_ALERT * 1.2)
        ax_gauge.set_yticks([])
        ax_gauge.set_title(f'σ Gauge (Current={sigma_est:.2f})')

        ax_hist.clear()
        ax_hist.hist(sigma_history, bins=HIST_BINS, range=(0, SIGMA_ALERT), color='blue')
        ax_hist.set_xlim(0, SIGMA_ALERT)
        ax_hist.set_title('Histogram σ')

        fig.canvas.draw()
        plt.pause(0.001)

    # desenha Pygame
    screen.fill((0, 0, 0))
    screen.blit(camera_surface, (0, 0))

    # painel de parâmetros
    panel = pygame.Surface((300, 600)); panel.fill((30, 30, 30))
    y = 10
    panel.blit(font.render(f"μ: {mu_est:.2f}" if mu_est is not None else "μ: --", True, (255, 255, 255)), (10, y)); y += 30
    panel.blit(font.render(f"σ: {sigma_est:.2f}" if sigma_est is not None else "σ: --", True, (255, 255, 255)), (10, y)); y += 30
    panel.blit(font.render(f"α: {alpha_est:.2f}" if alpha_est is not None else "α: --", True, (255, 255, 255)), (10, y)); y += 40
    if sigma_est is not None and sigma_est > SIGMA_ALERT:
        panel.blit(font.render("! ANOMALY !", True, (255, 0, 0)), (10, y)); y += 40

    screen.blit(panel, (800, 0))
    pygame.display.flip()

    # eventos Pygame
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            running = False
            break

    clock.tick(20)

# limpeza
logf.close()
pygame.quit()
imu_sensor.destroy()
camera.destroy()
for v in all_vehicles:
    v.destroy()