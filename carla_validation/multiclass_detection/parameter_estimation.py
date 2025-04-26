# parameter_estimation.py
# ------------------------
# Estima parâmetros AR(1) por classe usando janelas deslizantes rotuladas por maioria.

import os
import json
import numpy as np
from sklearn.model_selection import KFold
from collections import defaultdict, Counter

# Configurações
WINDOW_SIZE = 50    # comprimento da janela
N_ITER = 5          # iterações para MLE
KFOLDS = 5          # número de folds para validação

# Diretórios
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
DATA_IMU_DIR = os.path.join(PROJECT_ROOT, 'data', 'training', 'imu')
LABELS_CSV   = os.path.join(PROJECT_ROOT, 'data', 'training', 'labels', 'labels.csv')

# Função de estimação AR(1) via MLE
def estimate_parameters(X, n_iter=N_ITER):
    mu = np.nanmean(X[1:])
    alpha = 0.0
    for _ in range(n_iter):
        num = np.nansum((X[1:] - mu) * X[:-1])
        den = np.nansum((X[:-1] - mu) ** 2)
        alpha = 0.0 if den == 0 else num / den
        mu = np.nanmean(X[1:] - alpha * X[:-1])
    res = X[1:] - (mu + alpha * X[:-1])
    sigma = np.sqrt(np.nanmean(res**2))
    return mu, sigma, alpha

# 1) Carrega rótulos e dados
datasets = []  # (frame_id, filepath, label)
with open(LABELS_CSV, 'r') as f:
    next(f)
    for line in f:
        fname, lbl = line.strip().split(',')
        fp = os.path.join(DATA_IMU_DIR, fname)
        frame_id = int(fname.split('_')[-1].split('.')[0])
        datasets.append((frame_id, fp, lbl))
datasets.sort(key=lambda x: x[0])  # ordena por frame

# Monta arrays de magnitude e label por frame
series = []
labels_series = []
for _, fp, lbl in datasets:
    data = np.load(fp)
    mag = float(np.linalg.norm(data['accelerometer']))
    series.append(mag)
    labels_series.append(lbl)

# 2) Gera janelas deslizantes e atribui label por maioria
class_windows = defaultdict(list)
N = len(series)
for i in range(N - WINDOW_SIZE + 1):
    window = np.array(series[i:i + WINDOW_SIZE])
    lbls = labels_series[i:i + WINDOW_SIZE]
    # label da janela: valor mais frequente
    most_common = Counter(lbls).most_common(1)[0][0]
    class_windows[most_common].append(window)

# 3) Estima parâmetros por classe (com CV quando possível)
class_params = {}
total_windows = sum(len(w) for w in class_windows.values())
for lbl, windows in class_windows.items():
    arr = np.array(windows)
    n_win = len(arr)
    if n_win == 0:
        continue
    # se poucas janelas, estima tudo de uma vez
    if n_win < 2:
        est = np.array([estimate_parameters(w) for w in arr])
        mu_k, sigma_k, alpha_k = np.median(est, axis=0)
    else:
        splits = min(KFOLDS, n_win)
        if splits < 2: splits = 2
        kf = KFold(n_splits=splits, shuffle=True, random_state=42)
        mus, sigmas, alphas = [], [], []
        for train_idx, _ in kf.split(arr):
            tr = arr[train_idx]
            est = np.array([estimate_parameters(w) for w in tr])
            mus.append(np.median(est[:,0]))
            sigmas.append(np.median(est[:,1]))
            alphas.append(np.median(est[:,2]))
        mu_k = float(np.median(mus))
        sigma_k = float(np.median(sigmas))
        alpha_k = float(np.median(alphas))
    prior_k = n_win / total_windows
    class_params[lbl] = {'mu': mu_k, 'sigma': sigma_k, 'alpha': alpha_k, 'prior': prior_k}

# 4) Salva JSON
out_path = os.path.join(SCRIPT_DIR, 'class_params.json')
with open(out_path, 'w') as f:
    json.dump(class_params, f, indent=2)
print(f'Parâmetros salvos em {out_path}:', class_params)
