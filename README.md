[![DOI](https://zenodo.org/badge/967029477.svg)](https://doi.org/10.5281/zenodo.15674482)

# Language / Idioma
- [English](#english)
- [Português-BR](#português-br)

---
---

# English

# Temporal Naive Bayes (TNB) for Autonomous Navigation

This repository implements an adaptation of the classic Naive Bayes for sequential data, called **Temporal Naive Bayes (TNB)**. It is applied to real-time anomaly monitoring and multi-class maneuver classification using IMU signals in CARLA simulations, and successfully validated on real-world public smartphone datasets.

---

## 📋 Table of Contents

- [1. Overview](#1-overview)  
- [2. Mathematical Foundations](#2-mathematical-foundations)  
- [3. Experimental Pipeline](#3-experimental-pipeline)  
  - 3.1. Synthetic Data Generation  
  - 3.2. Offline Data Collection in CARLA  
  - 3.3. Automatic Labeling  
  - 3.4. Offline Parameter Estimation  
  - 3.5. Real-Time Integration  
  - 3.6. Multi-Class Classification  
  - 3.7. Anomaly Detection in Critical Scenarios  
  - 3.8. Real-World Smartphone Validation  
- [4. Folder Structure](#4-folder-structure)  
- [5. Installation & Dependencies](#5-installation--dependencies)  
- [6. Running the Scripts](#6-running-the-scripts)  
- [7. Usage Examples](#7-usage-examples)  
- [8. Publication](#8-publication)

---

## 1. Overview

Detecting anomalies and classifying maneuvers in autonomous robots/vehicles requires handling the **temporal dependency** in sensor data. Classic **Naive Bayes** assumes feature independence, which breaks down for time series. In TNB, we model each reading conditionally on the previous one using a **first-order autoregressive (AR(1))** model. This results in an interpretable, mathematically sound, and extremely lightweight algorithm ($>1000$ Hz on standard CPUs), ideal for edge devices.

---

## 2. Mathematical Foundations

### 2.1. Classic Naive Bayes

For classification into classes $C$:

$$P(C \mid X) \;\propto\; P(C)\,\prod_{i=1}^n P(x_i \mid C)$$

### 2.2. Temporal Naive Bayes (TNB)

We incorporate first-order dependence between sequential readings:

$$P(C \mid X_{1:T}) \;\propto\; P(C)\,\prod_{t=2}^T P\bigl(x_t \mid C,\,x_{t-1}\bigr)$$

### 2.3. AR(1) Model and Maximum Likelihood Estimation

We model each acceleration-magnitude window $x_t$ as:

$$x_t = \mu + \alpha\,x_{t-1} + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0,\sigma^2).$$

Maximizing the likelihood yields the closed-form MLE estimates:
1. $\displaystyle \hat\alpha = \frac{\sum_{t=2}^T (x_t-\mu)\,x_{t-1}}{\sum_{t=2}^T x_{t-1}^2}$
2. $\displaystyle \hat\mu = \frac{1}{T-1}\sum_{t=2}^T (x_t - \hat\alpha\,x_{t-1})$
3. $\displaystyle \hat\sigma = \sqrt{\frac{1}{T-1}\sum_{t=2}^T (x_t - \hat\mu - \hat\alpha x_{t-1})^2}$

The code implements these steps iteratively in the `parameter_estimation.py` functions.

---

## 3. Experimental Pipeline

### 3.1. Synthetic Data Generation
- Simulate AR(1) series with different parameters for two classes (Gaussian and Laplace noise).
- Evaluate accuracy sensitivity versus $\alpha$ and $\sigma$.
- **Notebooks:** `Ideal_Synthetic_TNB.ipynb`, `Realistic_Synthetic_TNB.ipynb`.

### 3.2. Offline Data Collection in CARLA
- Run CARLA in synchronous mode, collect IMU and vehicle control.

### 3.3. Automatic Labeling
`auto_label.py` reads control signals and labels each frame (`brake`, `turn`, `cruise`, `idle`) to generate a reproducible `labels.csv`.

### 3.4. Offline Parameter Estimation
Estimates $(\mu_k,\sigma_k,\alpha_k)$ per class via MLE + Stratified K-Fold Cross-Validation, saving to `class_params.json`.

### 3.5. Real-Time Integration
`real_time_multiclass.py` collects IMU into a sliding buffer, estimates parameters, computes log-likelihoods, and displays the prediction on a Pygame live dashboard.

### 3.6. Multi-Class Classification
- Standard benchmark in CARLA against GaussianNB, HMM, and One-Class SVM (`carla_benchmark_results.ipynb`).

### 3.7. Anomaly Detection in Critical Scenarios
- Dedicated scenario forcing a vehicle collision in CARLA to evaluate pure anomaly detection.
- We monitor the $\hat\sigma$ parameter against a $\sigma_{\mathrm{alert}}$ threshold.
- Compares TNB response against standard One-Class SVM (`anomaly_evaluation.ipynb`).

### 3.8. Real-World Smartphone Validation
- Benchmarks TNB on a publicly available 50 Hz Smartphone IMU dataset (`data/Daywise data/`).
- Validates the model's ability to classify aggressive vs. normal driving dynamics ("jerk") on physical edge hardware without simulator biases (`public_dataset_benchmark.ipynb`).

---

## 4. Folder Structure

```text
├── TNB-Autonomous-Navigation/
    ├── carla_validation/
    │   ├── anomaly_evaluation.ipynb
    │   ├── carla_benchmark_results.ipynb
    │   ├── collect_anomaly_data.py
    │   ├── real_time_tnb_integration.py
    │   └── multiclass_detection/
    │       ├── auto_label.py
    │       ├── multiclass_detection.py
    │       ├── parameter_estimation.py
    │       └── real_time_multiclass.py                  
    ├── data/
    │   └── Daywise data/           # Real-World Smartphone Sensor Dataset (Day-1 to Day-7)
    └── notebooks/
        ├── Ideal_Synthetic_TNB.ipynb
        ├── Realistic_Synthetic_TNB.ipynb
        └── public_dataset_benchmark.ipynb
```

---

## 5. Installation & Dependencies

```bash
# (Optional) Create the virtual environment (Recommended: conda-forge for Python 3.8+)
conda create -n tnb -c conda-forge python=3.10
conda activate tnb

# Install required packages
pip install numpy scipy pandas scikit-learn pygame matplotlib jupyter python-pptx carla
```
**Note:** Ensure the CARLA server is running on `localhost:2000` for simulation scripts.

---

## 6. Running the Scripts

### 6.1 CARLA Data Collection & Labeling
```bash
python carla_validation/multiclass_detection/data_collection.py
python carla_validation/multiclass_detection/auto_label.py
```

### 6.2 Parameter Estimation & Real-Time Demo
```bash
python carla_validation/multiclass_detection/parameter_estimation.py
python carla_validation/multiclass_detection/real_time_multiclass.py
```

### 6.3 Run Jupyter Notebooks (Benchmarks)
```bash
jupyter notebook
# Then open notebooks/public_dataset_benchmark.ipynb or carla_validation/anomaly_evaluation.ipynb
```

---

## 7. Usage Examples
- Adjust thresholds in `auto_label.py` for different scenarios.
- Tweak `WINDOW_SIZE` and `KFOLDS` in `parameter_estimation.py` to optimize performance.
- Observe the $\hat\sigma$ gauge in the Pygame dashboard during crashes to view anomaly triggering.

---

## 8. Publication
- Papers and further publication details are **under review**.

## 📝 License
This project is licensed under the MIT License.  
See the [LICENSE](./LICENSE) file for more details.

---
---

# Português-BR

# Temporal Naive Bayes (TNB) para Navegação Autônoma

Este repositório implementa uma adaptação do Naive Bayes clássico para dados sequenciais, denominada **Temporal Naive Bayes (TNB)**. É aplicado ao monitoramento de anomalias em tempo real e classificação multiclasse usando sinais de IMU no CARLA, e validado com sucesso em datasets públicos reais de smartphones.

---

## 📋 Sumário

- [1. Visão Geral](#1-visão-geral)  
- [2. Fundamentos Matemáticos](#2-fundamentos-matemáticos)  
- [3. Pipeline de Experimentação](#3-pipeline-de-experimentação)  
  - 3.1. Geração de Dados Sintéticos  
  - 3.2. Coleta Offline no CARLA  
  - 3.3. Rotulagem Automática  
  - 3.4. Estimação Offline de Parâmetros  
  - 3.5. Integração em Tempo Real  
  - 3.6. Classificação Multiclasse  
  - 3.7. Detecção de Anomalias em Cenários Críticos  
  - 3.8. Validação em Dataset Real (Smartphone)  
- [4. Estrutura de Pastas](#4-estrutura-de-pastas)  
- [5. Instalação e Dependências](#5-instalação-e-dependências)  
- [6. Execução dos Scripts](#6-execução-dos-scripts)  
- [7. Exemplos de Uso](#7-exemplos-de-uso)  
- [8. Publicação](#8-publicação)

---

## 1. Visão Geral

Detecção de anomalias e classificação de manobras em veículos autônomos requer o tratamento da **dependência temporal**. O **Naive Bayes** clássico falha para séries temporais pois assume independência. No TNB, modelamos cada leitura condicionalmente à anterior, usando um **modelo autorregressivo de primeira ordem (AR(1))**. O resultado é um algoritmo leve ($>1000$ Hz em CPUs padrão), interpretável e ideal para hardware de borda (Edge AI).

---

## 2. Fundamentos Matemáticos

*(As equações matemáticas são idênticas à seção em inglês acima. O modelo utiliza MLE em formato fechado para extrair estimativas de $\hat\mu$, $\hat\alpha$ e $\hat\sigma$).*

---

## 3. Pipeline de Experimentação

### 3.1. Geração de Dados Sintéticos
- Avalia sensibilidade matemática vs. $\alpha$ e $\sigma$ sob ruídos Gaussianos e de Laplace (`Ideal_Synthetic_TNB.ipynb` e `Realistic_Synthetic_TNB.ipynb`).

### 3.2. Coleta Offline no CARLA
- Executa simulação em modo síncrono, coletando IMU e controle do veículo.

### 3.3. Rotulagem Automática
Script `auto_label.py` lê controles e classifica frames em `brake`, `turn`, `cruise` ou `idle`.

### 3.4. Estimação Offline de Parâmetros
Estima parâmetros via MLE + K-Fold Cross Validation Estratificado.

### 3.5. Integração em Tempo Real
Script `real_time_multiclass.py` faz inferência em tempo real via *log-verossimilhança* com dashboard em Pygame.

### 3.6. Classificação Multiclasse
- Comparação contra GaussianNB, HMM e One-Class SVM em rotas dinâmicas (`carla_benchmark_results.ipynb`).

### 3.7. Detecção de Anomalias em Cenários Críticos
- Monitoramento direto do parâmetro estimado $\hat\sigma$ contra um limiar crítico de alerta.
- Inclui cenário de colisão forçada comparando a eficiência do TNB contra o One-Class SVM tradicional (`anomaly_evaluation.ipynb`).

### 3.8. Validação em Dataset Real (Smartphone)
- Uso de dataset público capturado a 50 Hz em tráfego real (`data/Daywise data/`).
- Isola e comprova a eficácia do parâmetro temporal $\alpha$ contra o modelo Gaussiano clássico na diferenciação de motoristas agressivos vs. normais (`public_dataset_benchmark.ipynb`).

---

## 4. Estrutura de Pastas
*(Veja a árvore de diretórios na seção em inglês acima).*

---

## 5. Instalação e Dependências

```bash
# (Opcional) Crie o ambiente virtual
conda create -n tnb -c conda-forge python=3.10
conda activate tnb

# Instalação de pacotes
pip install numpy scipy pandas scikit-learn pygame matplotlib jupyter python-pptx carla
```
**Nota:** Certifique-se de que o CARLA esteja rodando em `localhost:2000`.

---

## 6. Execução dos Scripts

### 6.1 Coleta e Rotulagem (CARLA)
```bash
python carla_validation/multiclass_detection/data_collection.py
python carla_validation/multiclass_detection/auto_label.py
```

### 6.2 Estimativa e Demo em Tempo Real
```bash
python carla_validation/multiclass_detection/parameter_estimation.py
python carla_validation/multiclass_detection/real_time_multiclass.py
```

### 6.3 Rodar Benchmarks em Jupyter
```bash
jupyter notebook
# Abra os arquivos .ipynb na pasta notebooks/ ou carla_validation/
```

---

## 7. Exemplos de Uso
- Ajuste thresholds em `auto_label.py` para cenários variados.
- Teste colisões simuladas no CARLA e veja a reação em tempo real do painel `real_time_multiclass.py`.

---

## 8. Publicação
- Artigos e detalhes de publicação: **EM REVISÃO**.

## 📝 Licença
Este projeto está licenciado sob a Licença MIT.  
Consulte o arquivo [LICENSE](./LICENSE) para detalhes.