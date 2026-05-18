# Reporte Técnico: Detector de Iris y Reconocimiento de Ojos

Este documento es un reporte técnico exhaustivo sobre el funcionamiento interno del proyecto. Detalla la arquitectura de las soluciones implementadas y cómo cada script contribuye al flujo del programa.

## 🧠 1. Arquitectura de Inteligencia Artificial (Red Neuronal)

El proyecto utiliza una **Red Neuronal Convolucional (CNN)** diseñada a la medida en PyTorch. Su función principal es la clasificación binaria: determinar si una imagen dada contiene un ojo (`Clase 0`) o no contiene un ojo (`Clase 1`).

### Flujo de la Información en la CNN (`model.py`)
1. **Entrada:** La imagen se transforma obligatoriamente a **Escala de Grises (1 canal)** y se redimensiona a **64x64 pixeles**. Esto se hizo para que el modelo no se vuelva dependiente de colores específicos de piel o iluminación, centrándose puramente en las texturas y geometrías.
2. **Capas Convolucionales (Feature Extraction):**
   - El modelo pasa por **3 filtros (Conv2D)**.
   - Cada filtro está seguido por una función de activación **ReLU** (que apaga los valores negativos) y una capa de agrupación **MaxPool2D** (que reduce la imagen a la mitad reteniendo lo más importante).
   - Resultado: De una imagen 64x64, el sistema destila "mapas de características" muy profundos de 8x8 pixeles.
3. **Cerebro Decisor (Capas Lineales):**
   - Los mapas se aplanan (`nn.Flatten`) en un vector lineal.
   - Pasan por una capa de 128 neuronas.
   - Se usa la técnica de **Dropout (50%)** durante el entrenamiento para "apagar" neuronas al azar. Esto fuerza a la red a no memorizar las fotos exactas, obligándola a aprender patrones reales (evita el *sobreajuste* o *overfitting*).
   - Finalmente, se colapsa a **1 neurona**. Usando la función matemática *Sigmoide*, esta neurona nos da un valor entre 0 y 1 representando el porcentaje de probabilidad.

## 🎯 2. Arquitectura de Visión Clásica (Localización del Iris)

Una vez que la red neuronal dice "Sí, esto es un ojo", entra a actuar la visión por computadora clásica en `analyze_iris.py`.

### ¿Cómo encuentra la pupila/iris?
En lugar de entrenar otra IA costosa para ubicar puntos en la pantalla (como lo haría un algoritmo YOLO o MediaPipe), este proyecto usa matemáticas simples pero eficientes:
1. Toma la imagen y la convierte a escala de grises.
2. Calcula dinámicamente un tamaño de **Filtro Gaussiano (Gaussian Blur)** basado en la resolución original (el núcleo de difuminado es siempre el 10% del tamaño de la imagen). 
3. Al aplicar este difuminado dinámico fuerte, texturas delgadas como las *pestañas* o *sombras pequeñas* desaparecen y se mezclan con la piel. Lo único que sobrevive es la masa oscura más grande del ojo (el iris).
4. El script hace un recorte lógico (ignora el 15% de los bordes perimetrales de la imagen) para evitar que el fondo negro, marcos, o pelo a los lados den falsos positivos.
5. Finalmente, busca **el valor mínimo matemático** (el pixel más negro) de la imagen resultante en la región central, y devuelve las coordenadas X e Y corrigiendo los márgenes. ¡Pura brillantez matemática!

---

## 📂 3. Diccionario de Scripts: ¿Qué hace cada cosa?

Aquí se explica para qué sirve cada archivo dentro de la carpeta `src/`.

### 🗃️ Manejo de Datos
- **`dataset.py`:** Es el "albañil" de los datos. Entra a tus carpetas en `data/train`, lee todas las imágenes y las convierte en "Tensores" (matrices numéricas) de 64x64 pixeles, escala de grises y las normaliza para que la red neuronal las pueda consumir en pequeños lotes de 32 a la vez.
- **`generate_synthetic_data.py`:** Como a veces no hay miles de fotos reales de ojos disponibles para empezar, este script usa lógica de dibujo para crear "ojos falsos" (óvalos con un punto negro) y "no ojos" (cuadrados aleatorios) para poder tener datos iniciales con los que la red empiece a aprender a diferenciar formas base.

### 🏋️ Entrenamiento del Modelo
- **`train.py`:** Es la "escuela" de la IA. Importa el modelo vacío (`model.py`) y las imágenes transformadas (`dataset.py`). Juega al ensayo y error: le pregunta al modelo qué es, lo castiga matemáticamente (usando Binary Cross Entropy) si se equivoca, y el optimizador Adam ajusta los engranajes. Al final de 10 vueltas (épocas), guarda su aprendizaje en el archivo `models/mi_modelo_detectar_ojos.pth`.
- *(También contiene lógica de "Fine-tuning" para reentrenarse si la corriges mediante la interfaz gráfica).*
- **`train_fast.py`:** Una versión recortada o de pruebas rápidas.

### 👁️ Evaluación y Uso en el Mundo Real
- **`predict.py`:** Lo ejecutas en tu terminal pasándole la ruta de 1 sola imagen. Es un script simple que te dice en texto "Es ojo" o "No es ojo".
- **`test_testing_folder.py`:** ¡Este es el script que acabamos de crear! Busca automáticamente todas las fotos en la carpeta global `data/testing`, las pasa por el escrutinio de la red y, si es un ojo, le ubica el iris con la matemática gaussiana. Muestra todo junto en una parrilla bonita (gracias a `matplotlib`).
- **`show_eyes_and_iris.py`:** Idéntico al anterior, pero en vez de probar fotos libres, agarra las mismas fotos con las que entrenó para verificar visualmente que sí haya aprendido (útil para auditoría interna).

### 🖥️ Interfaz de Interacción Humana
- **`gui_validator.py`:** Levanta una aplicación de escritorio con *Tkinter*. Te muestra imágenes, te pone una cruz roja donde cree que está el iris y te dice lo que la red neuronal predice. **Lo importante aquí son los botones**. Te permite hacer clic en "Correcto" o "Incorrecto". 
- Si haces clic, guarda tu voto en un archivo `feedback_labels.csv`. Tiene un botón mágico para **reentrenar**. Si la IA falló mucho, usará tus correcciones (etiquetas manuales) para hacer un mini-entrenamiento rápido y corregir ese fallo de ahí en adelante. Es un sistema continuo de aprendizaje por retroalimentación humana (RLHF básico).

---

## 🚀 Conclusión
El proyecto es un excelente punto intermedio entre algoritmos clásicos y redes neuronales profundas (Deep Learning). Demuestra habilidades en la normalización de datos (evitar bugs con imágenes a color vs blanco y negro), arquitecturas eficientes (uso de convolución y dropout para no sobreajustar) y diseño de ciclo cerrado de retroalimentación (poder corregir los errores en vivo con un GUI).
