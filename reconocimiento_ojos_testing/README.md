# Detector de Ojos e Iris en Tiempo Real (Camara en Vivo)

Este proyecto extiende el modelo de clasificacion de ojos entrenado en `reconocimiento_ojos/` y lo lleva al mundo real: **abre la camara de tu computadora y detecta tus ojos e iris en tiempo real**.

## ¿Que hace exactamente?

Cuando ejecutas el programa, se abre una ventana con el video de tu camara. El sistema analiza cada frame (imagen) y ejecuta 4 pasos en cascada:

```
Frame de Camara
      |
      v
 [1] Haar Cascade (OpenCV) --> Detecta la CARA en el frame
      |
      v
 [2] Haar Cascade (OpenCV) --> Dentro de la cara, detecta los OJOS
      |
      v
 [3] Red Neuronal CNN (PyTorch) --> Confirma si el recorte realmente ES un ojo
      |
      v
 [4] Gaussian Blur + Min Pixel --> Localiza el CENTRO del iris/pupila
```

### Paso 1: Deteccion de Cara (Haar Cascade)
OpenCV incluye un detector de caras preentrenado llamado **Haar Cascade**. Este es un algoritmo clasico de vision por computadora (no es deep learning) que usa patrones de contraste blanco/negro para encontrar formas que parecen una cara humana. Lo usamos para acotar la zona de busqueda y no perder tiempo analizando el fondo.

### Paso 2: Deteccion de Ojos (Haar Cascade)
Dentro del rectangulo de la cara detectada, aplicamos **otro Haar Cascade especializado en ojos** (`haarcascade_eye.xml`). Este detector busca patrones tipicos de la region ocular: cejas, parpados y la forma oval del ojo. Nos da rectangulos con las zonas candidatas a ser ojos.

### Paso 3: Confirmacion con Red Neuronal CNN (Nuestro Modelo)
Aqui entra la inteligencia artificial que entrenamos nosotros. Tomamos cada recorte candidato y lo pasamos por nuestra **Red Neuronal Convolucional** (archivo `mi_modelo_detectar_ojos.pth`). La red toma la imagen, la convierte a escala de grises de 64x64 pixeles, la pasa por 3 capas de filtros convolucionales y nos dice con un porcentaje de certeza si realmente es un ojo o si es un falso positivo (como una ceja o la nariz).

**¿Por que necesitamos este paso si ya tenemos Haar Cascade?**
Porque Haar Cascade es rapido pero impreciso. Puede confundir cejas, reflejos de lentes o sombras con ojos. Nuestra CNN actua como un "segundo par de ojos" (ironia incluida) que filtra los errores.

### Paso 4: Localizacion del Iris (Gaussian Blur)
Una vez confirmado que es un ojo, aplicamos el algoritmo matematico clasico:
- Convertimos el recorte del ojo a escala de grises.
- Aplicamos un desenfoque gaussiano con kernel dinamico (10% del tamanio de la imagen).
- Ignoramos el 15% de los bordes para evitar que pestanias o marcos interfieran.
- Buscamos el pixel mas oscuro de la region central. Ese punto es el centro de la pupila/iris.
- Dibujamos una cruz roja y un circulo amarillo en ese punto exacto.

---

## Estructura del Proyecto

```
reconocimiento_ojos_testing/
|
|-- src/
|   |-- detector_camara.py    # Script principal (abre camara y ejecuta todo)
|
|-- capturas/                  # Capturas de pantalla guardadas con [S]
|
|-- requirements.txt           # Dependencias de Python
|-- README.md                  # Este archivo
```

**Dependencia del proyecto original:** Este proyecto importa el modelo `DetectorDeOjosCNN` directamente desde `reconocimiento_ojos/src/model.py` y carga los pesos entrenados desde `reconocimiento_ojos/models/mi_modelo_detectar_ojos.pth`. No necesitas copiar nada, el script lo encuentra automaticamente.

---

## Como Ejecutarlo

### 1. Asegurate de tener las dependencias
```bash
pip install -r requirements.txt
```

### 2. Asegurate de haber entrenado el modelo
Si aun no has entrenado la red neuronal, ve a la carpeta del proyecto original y ejecuta:
```bash
cd reconocimiento_ojos
python src/train.py
```

### 3. Ejecuta el detector de camara
```bash
python reconocimiento_ojos_testing/src/detector_camara.py
```

### 4. Controles en la ventana
| Tecla | Accion |
|-------|--------|
| **Q** o **ESC** | Cerrar la camara y salir |
| **S** | Guardar una captura de pantalla en `capturas/` |

---

## Tecnologias Utilizadas

| Tecnologia | Uso en el proyecto |
|------------|-------------------|
| **OpenCV (cv2)** | Acceso a la camara, Haar Cascade para deteccion de cara/ojos, dibujo de graficos sobre el video |
| **PyTorch / Torchvision** | Cargar nuestra Red Neuronal CNN entrenada para confirmar si es ojo o no |
| **NumPy** | Manipulacion de matrices de pixeles del frame de video |
| **Haar Cascade** | Algoritmo clasico de OpenCV que detecta patrones de contraste (cara, ojos) sin deep learning |

## Flujo Visual

```
Camara USB/Integrada
       |
       v
  OpenCV captura frame BGR (ej. 640x480)
       |
       v
  cv2.cvtColor -> Escala de grises
       |
       v
  Haar Cascade Cara -> Rectangulo(s) de caras
       |
       v
  Haar Cascade Ojos -> Rectangulo(s) de ojos (dentro de cada cara)
       |
       v
  Recorte del ojo -> transforms.Grayscale + Resize 64x64 + Normalize
       |
       v
  CNN Forward Pass -> Sigmoid -> prob < 0.5? -> ES OJO
       |
       v
  GaussianBlur + argmin -> (x_iris, y_iris) -> Cruz roja + Circulo amarillo
       |
       v
  cv2.imshow -> Mostrar frame con anotaciones en pantalla
```

## Diferencias con el Proyecto Original

| Aspecto | reconocimiento_ojos | reconocimiento_ojos_testing |
|---------|--------------------|-----------------------------|
| Entrada | Imagenes estaticas (archivos .jpg/.png) | Video en tiempo real (camara) |
| Deteccion de zona | Manual (carpetas organizadas) | Automatica (Haar Cascade encuentra la cara y ojos) |
| Velocidad | Sin restriccion de tiempo | Debe procesar ~30 frames/segundo |
| Salida | Graficas de matplotlib | Ventana de video con anotaciones en vivo |
