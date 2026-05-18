# 👁️ Eyestim — Detector de Ojos e Iris con Inteligencia Artificial

Sistema de visión por computadora que **detecta ojos**, **clasifica si una imagen contiene un ojo** y **localiza el centro exacto del iris/pupila**, combinando Redes Neuronales Convolucionales (CNN) con algoritmos clásicos de procesamiento de imágenes.

Incluye un módulo de **detección en tiempo real** que abre la cámara de la computadora y analiza los ojos del sujeto en vivo.

---

## 🏗️ Estructura del Repositorio

```
Eyestim/
│
├── reconocimiento_ojos/          # Módulo 1: Entrenamiento y Clasificación
│   ├── src/                      # Código fuente (CNN, entrenamiento, predicción)
│   ├── docs/                     # Documentación técnica detallada
│   ├── data/                     # Imágenes de entrenamiento (no incluidas en el repo)
│   ├── models/                   # Modelos entrenados .pth (no incluidos en el repo)
│   └── README.md                 # Documentación del módulo
│
├── reconocimiento_ojos_testing/  # Módulo 2: Detección en Tiempo Real (Cámara)
│   ├── src/                      # Script de cámara en vivo
│   └── README.md                 # Documentación del módulo
│
└── README.md                     # Este archivo
```

---

## ⚙️ ¿Cómo Funciona?

El proyecto trabaja en **dos fases** que se complementan:

### Fase 1 — Clasificación con Red Neuronal CNN
Una Red Neuronal Convolucional entrenada desde cero con PyTorch determina si una imagen contiene un ojo o no.

```
Imagen de entrada (cualquier tamaño, color o B/N)
    ↓
Conversión a Escala de Grises (1 canal) + Resize 64x64
    ↓
3 Capas Convolucionales (Conv2D → ReLU → MaxPool)
    ↓
Capas Lineales + Dropout (50%)
    ↓
1 Neurona → Sigmoid → Probabilidad (0.0 a 1.0)
    ↓
¿Es un ojo? (prob < 0.5 = SÍ)
```

### Fase 2 — Localización del Iris (Visión Clásica)
Un algoritmo matemático eficiente ubica el centro exacto de la pupila sin necesidad de otra red neuronal.

```
Imagen del ojo → Escala de grises
    ↓
Gaussian Blur dinámico (kernel = 10% del tamaño de la imagen)
    ↓
Recorte central (ignora 15% de los bordes → evita pelo, cejas, marcos)
    ↓
Búsqueda del píxel más oscuro → Centro del iris (x, y)
```

### Fase 3 — Cámara en Tiempo Real (OpenCV)
Combina Haar Cascades + CNN + Gaussian Blur para analizar video en vivo:

```
Cámara → Haar Cascade (cara) → Haar Cascade (ojos) → CNN (confirma) → Iris (localiza)
```

---

## 🚀 Inicio Rápido

### 1. Clonar el repositorio
```bash
git clone https://github.com/Eddyfals0/Eyestim.git
cd Eyestim
```

### 2. Instalar dependencias
```bash
pip install torch torchvision Pillow matplotlib opencv-python numpy
```

### 3. Generar datos sintéticos (si no tienes imágenes reales)
```bash
cd reconocimiento_ojos
python src/generate_synthetic_data.py
```

### 4. Entrenar el modelo
```bash
python src/train.py
```

### 5. Probar con imágenes estáticas
```bash
python src/predict.py ruta/a/tu/imagen.jpg
```

### 6. Abrir la cámara en tiempo real
```bash
cd ..
python reconocimiento_ojos_testing/src/detector_camara.py
```
> Presiona **Q** para salir | **S** para guardar una captura

---

## 🛠️ Tecnologías

| Tecnología | Uso |
|------------|-----|
| **PyTorch** | Arquitectura CNN, entrenamiento y predicción |
| **OpenCV** | Acceso a cámara, Haar Cascades, procesamiento de video |
| **Pillow** | Carga y manipulación de imágenes estáticas |
| **Matplotlib** | Visualización de resultados y gráficas |
| **Tkinter** | Interfaz gráfica para validación humana (RLHF) |

---

## 📚 Documentación

Cada módulo tiene su propia documentación detallada:

- [`reconocimiento_ojos/README.md`](reconocimiento_ojos/README.md) — Arquitectura del modelo, guía de uso y explicación técnica
- [`reconocimiento_ojos/docs/REPORTE_TECNICO.md`](reconocimiento_ojos/docs/REPORTE_TECNICO.md) — Reporte profundo de cada script y su lógica interna
- [`reconocimiento_ojos_testing/README.md`](reconocimiento_ojos_testing/README.md) — Pipeline de cámara en tiempo real paso a paso

---

## 📄 Licencia

Proyecto académico universitario desarrollado con fines educativos.
