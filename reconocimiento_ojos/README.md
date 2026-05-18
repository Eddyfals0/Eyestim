# Detector de Iris y Reconocimiento de Ojos

Este proyecto es un sistema de visión por computadora que combina **Redes Neuronales Convolucionales (CNN)** con **procesamiento clásico de imágenes** para detectar si una imagen contiene un ojo y, de ser así, localizar el centro del iris.

## 🗂️ Estructura del Proyecto (Organización)

Para mantener el código ordenado, el proyecto está estructurado de la siguiente manera:

```text
reconocimiento_ojos/
│
├── data/                   # Contiene las imágenes para entrenar y validar el modelo (train/ val/)
│   └── feedback_labels.csv # Archivo generado por el validador GUI para reentrenamiento
│
├── docs/                   # Documentación adicional del proyecto
│
├── models/                 # Aquí se guardan los modelos entrenados (.pth)
│   └── mi_modelo_detectar_ojos.pth
│
├── results/                # Resultados y gráficas de evaluación generados por los scripts
│
├── src/                    # Código fuente principal
│   ├── model.py            # Arquitectura de la Red Neuronal (CNN)
│   ├── dataset.py          # Lógica para cargar y transformar imágenes
│   ├── train.py            # Script principal de entrenamiento y reentrenamiento
│   ├── train_fast.py       # Script de entrenamiento rápido/simplificado
│   ├── predict.py          # Script para predecir si una imagen individual es un ojo
│   ├── analyze_iris.py     # Lógica clásica para encontrar el centro del iris (Difuminado Gaussiano)
│   ├── show_eyes_and_iris.py # Genera matriz visual con predicción y centro del iris
│   ├── gui_validator.py    # Interfaz gráfica (Tkinter) para validar/corregir predicciones
│   └── generate_synthetic_data.py # Generador de datos sintéticos (dibujos) para pruebas
│
├── requirements.txt        # Dependencias de Python necesarias
└── README.md               # Este archivo de documentación
```

## ⚙️ ¿Cómo Funciona? (Arquitectura Técnica)

El proyecto ataca el problema en dos fases distintas, mezclando Deep Learning y Visión por Computadora clásica:

### 1. Inteligencia Artificial (Red Neuronal CNN)
La función principal de la red neuronal es la **clasificación binaria**: determina si la imagen contiene un ojo (`Clase 0`) o no (`Clase 1`).
- **Transformación Inicial:** Toda imagen es convertida a **escala de grises (1 canal)** y redimensionada a **64x64 pixeles**. Esto hace que el modelo no se vuelva dependiente del color de piel o iluminación, haciéndolo robusto y "daltónico", enfocándose solo en la textura y la forma.
- **Capas Convolucionales:** La imagen pasa por 3 filtros de convolución matemática, que van extrayendo características cada vez más complejas (bordes, curvas, formas esféricas). Tras pasar por funciones de reducción espacial (`MaxPool2d`), la imagen es destilada hasta sus patrones más puros.
- **Cerebro Decisor:** Los datos filtrados se aplanan y pasan por capas lineales. Una neurona final escupe la probabilidad (de 0.0 a 1.0) usando una función sigmoide.

### 2. Visión por Computadora (Localización Matemática del Iris)
En lugar de entrenar otra IA costosa para ubicar un punto exacto en la pantalla, utilizamos lógica eficiente implementada en `analyze_iris.py`.
- Toma la imagen transformada en escala de grises.
- Calcula dinámicamente un tamaño de **Filtro Gaussiano** que equivale al 10% de la imagen, borrando por completo cualquier detalle fino o ruido (pestañas).
- Aplica una **máscara de recorte central** (ignora el 15% de los bordes). Esto soluciona falsos positivos donde el cabello o los marcos oscuros de la foto distraían al algoritmo.
- Lo único que logra sobrevivir a tanto desenfoque en el centro es la masa oscura más grande del ojo: el Iris/Pupila.
- El algoritmo simplemente busca la coordenada (X, Y) del *píxel más oscuro de esa área* y lo mapea de vuelta. ¡Es magia matemática ultra rápida!

## 📂 Archivos Principales del Proyecto

- `src/model.py`: La arquitectura de la red (Filtros y Neuronas).
- `src/train.py`: Entrena la red desde cero o la actualiza basándose en feedback humano.
- `src/dataset.py`: Carga y transforma las fotos de `data/` al vuelo.
- `src/analyze_iris.py`: Encuentra la pupila usando el desenfoque Gaussiano.
- `src/test_testing_folder.py`: Analiza imágenes aleatorias externas colocadas en `data/testing/`.
- `src/gui_validator.py`: Interfaz de escritorio (RLHF). Te permite corregir a la IA visualmente para que se vuelva a reentrenar.

## 🚀 Guía de Uso

### 1. Instalación de dependencias
Asegúrate de instalar los requerimientos con:
```bash
pip install -r requirements.txt
```

### 2. Preparar los datos
Si no tienes datos reales aún, puedes generar imágenes sintéticas de prueba:
```bash
python src/generate_synthetic_data.py
```

### 3. Entrenar el modelo
Para que la red neuronal aprenda a distinguir ojos de otros objetos:
```bash
python src/train.py
```
*Esto generará el archivo `mi_modelo_detectar_ojos.pth` en la carpeta `models/`.*

### 4. Visualizar y Predecir
Para ver el sistema en acción evaluando múltiples imágenes y ubicando el iris:
```bash
python src/show_eyes_and_iris.py
```

Para predecir una sola imagen:
```bash
python src/predict.py ruta/a/tu/imagen.jpg
```

### 5. Validación Interactiva y Re-entrenamiento (Feedback Loop)
Puedes abrir una interfaz gráfica para corregir a la IA si se equivoca:
```bash
python src/gui_validator.py
```
Si corriges suficientes imágenes, el programa guardará un archivo `feedback_labels.csv` y te permitirá reentrenar la red neuronal con un solo botón en la interfaz (Fine-tuning).

## 🛠️ Tecnologías Usadas
- **PyTorch / Torchvision**: Para crear la Red Neuronal, entrenarla y aplicar tensores de transformación.
- **Pillow (PIL)**: Para carga y dibujado básico de imágenes.
- **Matplotlib**: Para graficar y mostrar resultados de la detección del iris y clasificación en matrices de fotos.
- **Tkinter**: Para la interfaz gráfica de usuario en la validación.
