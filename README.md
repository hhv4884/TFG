# Análisis de Poda y Cuantización en Redes Neuronales

Repositorio con la implementación utilizada para estudiar el efecto de distintas estrategias de poda (*pruning*) y cuantización sobre redes neuronales *feedforward* multicapa.

El proyecto incluye:
- entrenamiento de modelos,
- generación de datasets sintéticos,
- experimentos de poda y cuantización,
- agregación estadística,
- y visualización de resultados.

---

# Estructura del Repositorio

```text
.
├── core/
│   ├── accuracy.py
│   ├── load_dataset.py
│   ├── models.py
│   ├── new_dataset.py
│   ├── training.py
│   └── utils.py
│
├── datasets/
├── figures/
├── results/
├── training/
│
├── main.py
├── mainEXPERIMENT.py
├── mainPLT.py
├── mainPLTDATASETS.py
├── mainRESULTS.py
├── mainTRAIN.py
├── prueba.py
│
├── requirements.txt
└── README.md
```

---

# Core

## `accuracy.py`

### `obtein_accuracies(...)`
Calcula la precisión de un modelo bajo distintas estrategias de poda y cuantización.

#### Parámetros principales
- `train_loader`, `test_loader`: datasets.
- `base_path`: ruta donde guardar resultados.
- `input`: dimensión de entrada.
- `hidden_size`: tamaño de la capa oculta.
- `device`: CPU o GPU.

#### Estrategias evaluadas
- `smallest`
- `random`
- `highest`
- cuantización por signo (`to_signs=True`)

Los resultados se almacenan en `results_log.csv`.

---

## `load_dataset.py`

### `load_mnist(batch_size, classes)`
Carga MNIST filtrando únicamente las clases indicadas.

### `load_fashion_mnist(batch_size, classes)`
Carga Fashion-MNIST filtrando únicamente las clases indicadas.

### `load_base(task, n_samples, class_sep, cluster_number, noise, batch_size, n_features)`
Genera datasets sintéticos.

#### Tipos soportados
- `linear`
- `radial`
- `non_linear_non_radial`
- `classification`

#### Parámetros principales

- `class_sep`: controla la separación entre clases en tareas de clasificación (`make_classification`). Valores altos generan problemas más fáciles de separar.

- `cluster_number`: controla la dispersión de los clusters en datasets lineales (`make_blobs`). Valores altos generan mayor solapamiento entre clases.

- `noise`: añade ruido en datasets no lineales y radiales (`make_moons` y `make_circles`), aumentando la dificultad de clasificación.

- `n_features`: define la dimensionalidad de entrada del dataset (`make_classification` y `make_blobs`).

Devuelve `train_loader` y `test_loader`.

---

## `models.py`

### `MLP`
Implementa una red neuronal *feedforward* multicapa.

### `MLP_superior`
Wrapper principal del modelo utilizado durante los experimentos.

### `threshold_weight_forward(...)`
Aplica poda dinámica durante inferencia.

#### Parámetros principales
- `method`:
  - `smallest`
  - `highest`
  - `random`
- `fraction_non_zero`: porcentaje de pesos conservados.
- `to_signs`: aplica cuantización ternaria.

### `save_edgelist(...)`
Exporta pesos como lista de aristas.

### `save_biaslist(...)`
Exporta biases del modelo.

---

## `new_dataset.py`

### `NewDataset`
Wrapper personalizado para transformar problemas multiclase en clasificación binaria.

#### Funcionalidades
- filtrado de clases,
- remapeo de etiquetas,
- compatibilidad con PyTorch.

#### Ejemplos
- `[0,7] → [0,1]`
- `[2,3] → [0,1]`

---

## `training.py`

### `train_model_superior_without_early_stop(...)`
Entrena un modelo sin *early stopping*.

### `train_model_superior_with_early_stop(...)`
Entrena un modelo utilizando *early stopping*.

#### Configuración utilizada
- optimizador `Adam`
- `NLLLoss`
- scheduler `CosineAnnealingLR`

---

## `utils.py`

### `save_logs(epoch, path)`
Genera archivos de log por época.

### `clean_data(data, classes)`
Filtra clases concretas de un dataset.

### `get_epoch_from_folder(path)`
Obtiene la mejor época almacenada.

### `extract_test_acc_from_log(log_path)`
Extrae accuracy desde logs.

### `compute_medians_linear(...)`
### `compute_medians_radial(...)`
### `compute_medians_moons(...)`
Calculan medianas sobre múltiples ejecuciones.

### `load_model_from_txt(...)`
Reconstruye modelos desde archivos de pesos y biases.

### `compute_accuracy(...)`
Calcula accuracy bajo distintos métodos de inferencia.

#### Métodos soportados
- `base`
- `threshold`

---

# Scripts Principales

## `mainTRAIN.py`

Entrena todos los modelos utilizados en los experimentos.

### Experimentos incluidos
- MNIST
- Fashion-MNIST
- datasets de `make_blobs`
- datasets radiales
- datasets tipo moons
- dataset de `make_classification`

### Variables exploradas
- tamaños de red,
- ruido,
- separación entre clases,
- múltiples inicializaciones,
- *early stopping*.

---

## `mainEXPERIMENT.py`

Carga modelos entrenados y evalúa:
- poda,
- cuantización,

Genera automáticamente archivos `results_log.csv`.

---

## `mainRESULTS.py`

Agrega resultados de múltiples ejecuciones.

### Estadísticos calculados
- mediana,
- percentil 25,
- percentil 75.

Los resultados se almacenan en `results/`.

---

## `mainPLT.py`

Genera las gráficas finales de resultados.

### Comparaciones incluidas
- `smallest`
- `random`
- `largest`
- variantes cuantizadas
- *early stopping* vs no *early stopping*

Las figuras se almacenan en `figures/`.

---

## `mainPLTDATASETS.py`

Genera visualizaciones de datasets sintéticos.

### Datasets visualizados
- `make_classification`
- `make_blobs`

Las figuras se almacenan en `datasets/`.

---

# Datasets Utilizados

## Reales
- MNIST
- Fashion-MNIST

## Sintéticos
- `make_blobs`
- `make_circles`
- `make_moons`
- `make_classification`

---

# Ejecución

## Instalar dependencias

```bash
pip install -r requirements.txt
```

## Entrenar modelos

```bash
python mainTRAIN.py
```

## Ejecutar experimentos

```bash
python mainEXPERIMENT.py
```

## Agregar resultados

```bash
python mainRESULTS.py
```

## Generar gráficas

```bash
python mainPLT.py
```

## Visualizar datasets

```bash
python mainPLTDATASETS.py
```

---

# Tecnologías Utilizadas

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

# Autor

Lucía Campos Díez

Trabajo Fin de Grado: Analizando la estructura interna de
redes neuronales a través de técnicas
de cuantización y pruning.