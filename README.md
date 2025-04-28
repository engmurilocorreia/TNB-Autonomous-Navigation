# Temporal Naive Bayes (TNB) para Navegação Autônoma

Este repositório implementa uma adaptação do Naive Bayes clássico para dados sequenciais, denominada **Temporal Naive Bayes (TNB)**, aplicada à detecção de anomalias e classificação multiclasse usando sinais de IMU em simulações no CARLA.

---

## 📋 Sumário

- [1. Visão Geral](#1-visão-geral)  
- [2. Fundamentos Matemáticos](#2-fundamentos-matemáticos)  
  - 2.1. Naive Bayes Clássico  
  - 2.2. Temporal Naive Bayes  
  - 2.3. Modelo AR(1) e MLE  
- [3. Pipeline de Experimentação](#3-pipeline-de-experimentação)  
  - 3.1. Geração de Dados Sintéticos  
  - 3.2. Coleta Offline no CARLA  
  - 3.3. Rotulagem Automática  
  - 3.4. Estimação Offline de Parâmetros  
  - 3.5. Integração em Tempo Real  
  - 3.6. Classificação Multiclasse  
- [4. Estrutura de Pastas](#4-estrutura-de-pastas)  
- [5. Instalação e Dependências](#5-instalação-e-dependências)  
- [6. Execução dos Scripts](#6-execução-dos-scripts)  
- [7. Exemplos de Uso](#7-exemplos-de-uso)  
- [8. Publicação](#8-publicação)

---

## 1. Visão Geral

Detecção de anomalias e classificação de manobras em robôs/carros autônomos requer o tratamento da **dependência temporal** nos dados de sensores. O **Naive Bayes** clássico assume independência entre as features, o que falha para séries temporais. No TNB, modelamos cada leitura condicionalmente à anterior, usando um **modelo autorregressivo de primeira ordem (AR(1))**.

---

## 2. Fundamentos Matemáticos

### 2.1. Naive Bayes Clássico

Para classificação em classes $C$:

$$P(C \mid X) \;\propto\; P(C)\,\prod_{i=1}^n P(x_i \mid C)$$


### 2.2. Temporal Naive Bayes (TNB)

Incorporamos dependência de primeira ordem entre leituras sequenciais:

$$P(C \mid X_{1:T}) \;\propto\; P(C)\,\prod_{t=2}^T P\bigl(x_t \mid C,\, x_{t-1}\bigr)$$


### 2.3. Modelo AR(1) e Estimação MLE

Modelamos cada janela de magnitude de aceleração $x_t$ como:

$$x_t = \mu + \alpha\,x_{t-1} + \varepsilon_t,\quad \varepsilon_t\sim\mathcal{N}(0,\sigma^2).$$


A função de verossimilhança para uma janela $x_1,\dots,x_T$ é:

$$\mathcal{L}(\mu,\alpha,\sigma)
=\prod_{t=2}^T\frac{1}{\sigma\sqrt{2\pi}}
\exp\!\Bigl[-\frac{(x_t-\mu-\alpha x_{t-1})^2}{2\sigma^2}\Bigr].$$


Maximizando esta função obtêm-se as estimativas (MLE):
1. $\displaystyle \hat\alpha = \frac{\sum_{t=2}^T (x_t-\mu)\,x_{t-1}}{\sum_{t=2}^T x_{t-1}^2}$  
2. $\displaystyle \hat\mu = \frac{1}{T-1}\sum_{t=2}^T (x_t - \hat\alpha\,x_{t-1}) $
3. $\displaystyle \hat\sigma = \sqrt{\frac{1}{T-1}\sum_{t=2}^T (x_t - \hat\mu - \hat\alpha x_{t-1})^2}$

O código implementa estas etapas iterativamente (função `estimate_parameters`).

---

## 3. Pipeline de Experimentação

### 3.1. Geração de Dados Sintéticos
- Simula séries AR(1) com parâmetros distintos para duas classes.
- Avalia sensibilidade de acurácia vs. \(\alpha\) e \(\sigma\).

### 3.2. Coleta Offline no CARLA
- Executa simulação em modo síncrono, coleta IMU e controle do veículo.
- Salva `.npz` com:
  - `timestamp`, `accelerometer`, `gyroscope`, `compass`
  - `throttle`, `steer`, `brake`

### 3.3. Rotulagem Automática
Script `auto_label.py` lê controles e classifica cada frame:
- `brake` se `brake > θ_b`
- `turn` se `|steer| > θ_s`
- `cruise` se `throttle > θ_t`
- `idle` caso contrário

Gera `labels.csv` de forma reprodutível.

### 3.4. Estimação Offline de Parâmetros
Script `parameter_estimation.py`:
1. Carrega `labels.csv` e janelas de IMU.
2. Desliza janelas de tamanho fixo e atribui label pela **maioria**.
3. Estima $(\mu_k,\sigma_k,\alpha_k)$ por classe via MLE + Validação Cruzada.
4. Salva `class_params.json` com parâmetros e priors.

### 3.5. Integração em Tempo Real
Script `real_time_multiclass.py`:
1. Carrega `class_params.json`.
2. Coleta IMU + buffer deslizante.
3. Estima parâmetros da janela corrente.
4. Calcula log-verossimilhança para cada classe:

```math
\ell_k(X)= -\sum_{t=2}^T\frac{(x_t-\mu_k-\alpha_k x_{t-1})^2}{2\sigma_k^2} - (T-1)\ln(\sigma_k\sqrt{2\pi})
+ \ln P(C_k).
```
5. Classifica $\hat k=\arg\max_k \ell_k(X)$.
6. Exibe predição em painel Pygame + gráficos Matplotlib.

### 3.6. Classificação Multiclasse
- Classes: `idle`, `cruise`, `turn`, `brake`
- Fácil extensão para novas classes/manobras.

---

## 4. Estrutura de Pastas
```bash
├── TNB-Autonomous-Navigation/
      ├── carla_validation/
      │      ├── data/
      │      ├── multiclass_detection/
      │      │      ├── data/
      │      │      ├── auto_label.py
      │      │      ├── class_params.json
      │      │      ├── multiclass_detection.py
      │      │      ├── parameter_estimation.py
      │      │      └── real_time_multiclass.py                  
      │      ├── data_collection.py
      │      ├── offline_processing_imu.py
      │      ├── plot_imu_data.py
      │      ├── real_time_tnb_integration.py
      │      └── t_nb_offline_analysis.py
      └── notebooks/
            ├── Ideal_Synthetic_TNB.ipynb
            └── Realistic_Synthetic_TNB.ipynb
```

---

## 5. Instalação e Dependências
```bash
# Ambiente virtual opcional
conda create -n tnb python=3.8
conda activate tnb

# Instalação de pacotes
pip install numpy scipy scikit-learn pygame matplotlib python-pptx carla
```
**Nota:** Certifique-se de que o servidor CARLA esteja rodando em `localhost:2000`.

---

## 6. Execução dos Scripts

### 6.1 Coleta de Dados (Offline)
```bash
python multiclass_detection/data_collection.py
```

### 6.2 Rotulagem Automática
```bash
python multiclass_detection/auto_label.py
```

### 6.3 Estimativa de Parâmetros
```bash
python multiclass_detection/parameter_estimation.py
```

### 6.4 Demo Real-Time Multiclasse
```bash
python multiclass_detection/real_time_multiclass.py
```

---

## 7. Exemplos de Uso

- Ajuste thresholds em `auto_label.py` para cenários variados.

- Modifique `WINDOW_SIZE` e `KFOLDS` em `parameter_estimation.py` para otimização.

- Use `real_time_multiclass.py` para capturar vídeos do painel e demonstrar resultados.

---

## 8. Publicação

-  Artigos Publicados: **EM DESENVOLVIMENTO**.
