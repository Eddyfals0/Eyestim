"""
detector_camara.py - Detector de Ojos en Tiempo Real con Camara
================================================================
Este script abre la camara de la computadora y detecta los ojos del sujeto
en tiempo real. Combina 3 tecnologias:
  1. OpenCV Haar Cascade  -> Para localizar la region del ojo en el frame
  2. Nuestra CNN (PyTorch) -> Para confirmar que SI es un ojo (clasificacion)
  3. Gaussian Blur clasico -> Para ubicar el centro exacto del iris/pupila
"""
import os
import sys
import cv2
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np

# ── Importar nuestro modelo entrenado ──────────────────────────────────
# Agregamos la carpeta src del proyecto original al PATH de Python
# para poder importar model.py directamente
PROYECTO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROYECTO_ORIGINAL = os.path.join(PROYECTO_DIR, "..", "reconocimiento_ojos", "src")
sys.path.insert(0, PROYECTO_ORIGINAL)

from model import DetectorDeOjosCNN


# ══════════════════════════════════════════════════════════════════════
#                     FUNCIONES AUXILIARES
# ══════════════════════════════════════════════════════════════════════

def cargar_modelo():
    """Carga el modelo CNN entrenado desde el proyecto original."""
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = DetectorDeOjosCNN()

    # Buscar el archivo .pth en varias rutas posibles
    rutas_posibles = [
        os.path.join(PROYECTO_DIR, "..", "reconocimiento_ojos", "models", "mi_modelo_detectar_ojos.pth"),
        os.path.join(PROYECTO_DIR, "..", "models", "mi_modelo_detectar_ojos.pth"),
    ]

    modelo_cargado = False
    for ruta in rutas_posibles:
        if os.path.exists(ruta):
            modelo.load_state_dict(torch.load(ruta, map_location=dispositivo))
            modelo.to(dispositivo)
            modelo.eval()
            modelo_cargado = True
            print(f"Modelo CNN cargado desde: {ruta}")
            break

    if not modelo_cargado:
        print("[AVISO] No se encontro el modelo CNN entrenado.")
        print("La deteccion funcionara solo con Haar Cascade (sin confirmacion IA).")

    return modelo, dispositivo, modelo_cargado


def encontrar_centro_iris(imagen_bgr):
    """
    Encuentra el centro del iris usando Gaussian Blur + busqueda de minimo.
    Recibe una imagen BGR de OpenCV (recorte del ojo).
    Retorna (x, y) relativos al recorte.
    """
    gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gris.shape

    # Kernel dinamico (10% de la imagen, debe ser impar)
    k = int(min(h, w) * 0.10)
    if k % 2 == 0:
        k += 1
    k = max(3, k)

    # Aplicar desenfoque gaussiano fuerte
    difuminado = cv2.GaussianBlur(gris, (k, k), 0)

    # Ignorar 15% de los bordes (pelo, cejas, marcos)
    my = int(h * 0.15)
    mx = int(w * 0.15)
    region = difuminado[my:h - my, mx:w - mx]

    if region.size == 0:
        region = difuminado
        my, mx = 0, 0

    # Encontrar el pixel mas oscuro (pupila/iris)
    _, _, min_loc, _ = cv2.minMaxLoc(region)
    cx = min_loc[0] + mx
    cy = min_loc[1] + my

    return cx, cy


def clasificar_ojo_cnn(imagen_bgr, modelo, dispositivo):
    """
    Usa nuestra CNN para evaluar si el recorte realmente es un ojo.
    Retorna (es_ojo: bool, certeza: float).
    """
    transformacion = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    tensor = transformacion(imagen_bgr).unsqueeze(0).to(dispositivo)

    with torch.no_grad():
        salida = modelo(tensor)
        prob = torch.sigmoid(salida).item()

    # prob < 0.5 -> clase 0 (ojo)
    if prob < 0.5:
        return True, (1 - prob) * 100
    else:
        return False, prob * 100


# ══════════════════════════════════════════════════════════════════════
#                    BUCLE PRINCIPAL DE CAMARA
# ══════════════════════════════════════════════════════════════════════

def iniciar_camara():
    """Abre la camara y ejecuta la deteccion en tiempo real."""

    # 1. Cargar clasificadores Haar Cascade de OpenCV
    ruta_haar_cara = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    ruta_haar_ojos = cv2.data.haarcascades + "haarcascade_eye.xml"

    detector_cara = cv2.CascadeClassifier(ruta_haar_cara)
    detector_ojos = cv2.CascadeClassifier(ruta_haar_ojos)

    if detector_cara.empty() or detector_ojos.empty():
        print("[ERROR] No se pudieron cargar los clasificadores Haar Cascade.")
        return

    # 2. Cargar nuestro modelo CNN
    modelo, dispositivo, tiene_cnn = cargar_modelo()

    # 3. Abrir la camara (indice 0 = camara principal)
    camara = cv2.VideoCapture(0)

    if not camara.isOpened():
        print("[ERROR] No se pudo abrir la camara.")
        print("Asegurate de que tu computadora tiene una camara conectada.")
        return

    print("=" * 55)
    print("  DETECTOR DE OJOS E IRIS EN TIEMPO REAL")
    print("=" * 55)
    print("Controles:")
    print("  [Q] o [ESC] -> Salir")
    print("  [S]         -> Guardar captura de pantalla")
    print("=" * 55)

    capturas = 0

    while True:
        ret, frame = camara.read()
        if not ret:
            print("No se pudo leer el frame de la camara.")
            break

        # Voltear horizontalmente para efecto espejo
        frame = cv2.flip(frame, 1)

        # Convertir a escala de grises para Haar Cascade
        gris_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── PASO 1: Detectar caras ──────────────────────────────
        caras = detector_cara.detectMultiScale(
            gris_frame,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(80, 80)
        )

        for (cx, cy, cw, ch) in caras:
            # Dibujar rectangulo de la cara (azul)
            cv2.rectangle(frame, (cx, cy), (cx + cw, cy + ch), (255, 180, 0), 2)

            # Recortar solo la region de la cara para buscar ojos
            roi_gris = gris_frame[cy:cy + ch, cx:cx + cw]
            roi_color = frame[cy:cy + ch, cx:cx + cw]

            # ── PASO 2: Detectar ojos dentro de la cara ─────────
            ojos = detector_ojos.detectMultiScale(
                roi_gris,
                scaleFactor=1.1,
                minNeighbors=10,
                minSize=(25, 25)
            )

            for (ox, oy, ow, oh) in ojos:
                # Recortar la region del ojo
                recorte_ojo = roi_color[oy:oy + oh, ox:ox + ow]

                if recorte_ojo.size == 0:
                    continue

                # ── PASO 3: Confirmar con nuestra CNN ───────────
                es_ojo = True
                certeza = 100.0

                if tiene_cnn:
                    es_ojo, certeza = clasificar_ojo_cnn(recorte_ojo, modelo, dispositivo)

                if es_ojo:
                    # Dibujar rectangulo verde alrededor del ojo
                    cv2.rectangle(
                        roi_color,
                        (ox, oy), (ox + ow, oy + oh),
                        (0, 255, 0), 2
                    )

                    # ── PASO 4: Localizar el iris ───────────────
                    iris_x, iris_y = encontrar_centro_iris(recorte_ojo)

                    # Convertir coordenadas al frame completo
                    iris_global_x = cx + ox + iris_x
                    iris_global_y = cy + oy + iris_y

                    # Dibujar cruz roja en el iris
                    r = 8
                    cv2.line(frame,
                             (iris_global_x - r, iris_global_y),
                             (iris_global_x + r, iris_global_y),
                             (0, 0, 255), 2)
                    cv2.line(frame,
                             (iris_global_x, iris_global_y - r),
                             (iris_global_x, iris_global_y + r),
                             (0, 0, 255), 2)

                    # Dibujar circulo alrededor del iris
                    radio_iris = min(ow, oh) // 4
                    cv2.circle(frame,
                               (iris_global_x, iris_global_y),
                               radio_iris, (0, 255, 255), 1)

                    # Texto con porcentaje
                    texto = f"Ojo {certeza:.0f}%"
                    cv2.putText(frame, texto,
                                (cx + ox, cy + oy - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 1)
                else:
                    # No es ojo segun la CNN -> rectangulo rojo
                    cv2.rectangle(
                        roi_color,
                        (ox, oy), (ox + ow, oy + oh),
                        (0, 0, 255), 1
                    )

        # Panel informativo
        cv2.putText(frame, "Detector de Ojos e Iris - Tiempo Real",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        cv2.putText(frame, "[Q] Salir | [S] Captura",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1)

        # Mostrar el frame
        cv2.imshow("Detector de Ojos e Iris", frame)

        # Leer teclas
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q') or tecla == 27:  # Q o ESC
            break
        elif tecla == ord('s'):
            capturas += 1
            ruta_captura = os.path.join(
                PROYECTO_DIR, "capturas", f"captura_{capturas}.png"
            )
            os.makedirs(os.path.join(PROYECTO_DIR, "capturas"), exist_ok=True)
            cv2.imwrite(ruta_captura, frame)
            print(f"Captura guardada: {ruta_captura}")

    # Liberar recursos
    camara.release()
    cv2.destroyAllWindows()
    print("Camara cerrada correctamente.")


if __name__ == "__main__":
    iniciar_camara()
