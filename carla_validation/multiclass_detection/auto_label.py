import os, csv, numpy as np

# pasta onde o script vive
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# sobe duas pastas até o diretório raiz do projeto
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

# define caminhos absolutos
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'training', 'imu')
OUT_CSV = os.path.join(PROJECT_ROOT, 'data', 'training', 'labels', 'labels.csv')

#DATA_DIR = "data/training/imu"
#OUT_CSV  = "data/training/labels/labels.csv"

# thresholds (ajuste conforme seu experimento)
STEER_THR  = 0.1   # curva se |steer| > 0.1
BRAKE_THR  = 0.2   # frenagem se brake  > 0.2
THROTTLE_THR = 0.2 # cruzeiro se throttle > 0.2

files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".npz")])
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename","label"])
    for fn in files:
        data = np.load(os.path.join(DATA_DIR, fn))
        steer   = float(data["steer"])
        brake   = float(data["brake"])
        throttle= float(data["throttle"])
        # regra simples de prioridade
        if brake > BRAKE_THR:
            lbl = "brake"
        elif abs(steer) > STEER_THR:
            lbl = "turn"
        elif throttle > THROTTLE_THR:
            lbl = "cruise"
        else:
            lbl = "idle"
        writer.writerow([fn, lbl])
print("labels.csv gerado com", len(files), "frames")
