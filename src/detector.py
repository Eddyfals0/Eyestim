import os
import sys
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np

# Resolver rutas relativas absolutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_MODELO = os.path.join(BASE_DIR, "models", "mi_modelo_detectar_ojos.pth")

# Intentar importar el modelo CNN
try:
    from model import DetectorDeOjosCNN
    tiene_modulo_modelo = True
except ImportError:
    tiene_modulo_modelo = False

# ══════════════════════════════════════════════════════════════════════
#                     FUNCIONES AUXILIARES
# ══════════════════════════════════════════════════════════════════════

def cargar_modelo():
    """Carga el modelo CNN entrenado si existe en models/."""
    if not tiene_modulo_modelo:
        print("[AVISO] No se pudo importar DetectorDeOjosCNN. Ejecución sin confirmación IA.")
        return None, None, False

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = DetectorDeOjosCNN()

    if os.path.exists(RUTA_MODELO):
        try:
            modelo.load_state_dict(torch.load(RUTA_MODELO, map_location=dispositivo))
            modelo.to(dispositivo)
            modelo.eval()
            print(f"Modelo CNN cargado con éxito desde: {RUTA_MODELO}")
            return modelo, dispositivo, True
        except Exception as e:
            print(f"[AVISO] Error al cargar los pesos del modelo: {e}")
    else:
        print(f"[AVISO] No se encontró el modelo entrenado en '{RUTA_MODELO}'.")
        print("La detección funcionará con Haar Cascades de OpenCV sin filtrado CNN.")
    
    return None, None, False


def clasificar_ojo_cnn(imagen_bgr, modelo, dispositivo):
    """Evalúa si un recorte de imagen contiene un ojo usando la CNN."""
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

    # Mapeo: clase 0 (ojo) si prob < 0.5
    if prob < 0.5:
        return True, (1 - prob) * 100
    else:
        return False, prob * 100


def encontrar_centro_y_tamano_iris(recorte_ojo):
    """
    Encuentra el centro del iris (píxel más oscuro) y calcula el tamaño
    estimado de la retina/pupila analizando el gradiente de intensidad.
    
    Retorna:
        cx, cy (coordenadas del centro relativas al recorte)
        diametro_pupila (diámetro estimado de la pupila en píxeles)
    """
    gris = cv2.cvtColor(recorte_ojo, cv2.COLOR_BGR2GRAY)
    h, w = gris.shape

    # 1. Aplicar Gaussian Blur dinámico proporcional para suavizar ruido
    k = int(min(h, w) * 0.10)
    if k % 2 == 0:
        k += 1
    k = max(3, k)
    difuminado = cv2.GaussianBlur(gris, (k, k), 0)

    # 2. Ignorar el 15% de los bordes para evitar cejas y párpados oscuros
    my = int(h * 0.15)
    mx = int(w * 0.15)
    region = difuminado[my:h - my, mx:w - mx]

    if region.size == 0:
        region = difuminado
        my, mx = 0, 0

    # 3. Ubicar el centro del iris (el punto más oscuro de la región central)
    min_val, _, min_loc, _ = cv2.minMaxLoc(region)
    cx = min_loc[0] + mx
    cy = min_loc[1] + my

    # 4. Algoritmo experimental de cruz oscura para medir el tamaño de la pupila
    # Avanzamos radialmente en las 4 direcciones (+) hasta cruzar un umbral
    umbral = int(min_val + 20)
    
    # Derecha
    r_der = 0
    for x in range(cx, w):
        if difuminado[cy, x] > umbral:
            r_der = x - cx
            break
    if r_der == 0: r_der = w - cx

    # Izquierda
    r_izq = 0
    for x in range(cx, -1, -1):
        if difuminado[cy, x] > umbral:
            r_izq = cx - x
            break
    if r_izq == 0: r_izq = cx

    # Abajo
    r_abaj = 0
    for y in range(cy, h):
        if difuminado[y, cx] > umbral:
            r_abaj = y - cy
            break
    if r_abaj == 0: r_abaj = h - cy

    # Arriba
    r_arr = 0
    for y in range(cy, -1, -1):
        if difuminado[y, cx] > umbral:
            r_arr = cy - y
            break
    if r_arr == 0: r_arr = cy

    # El diámetro estimado es el promedio de los anchos en cruz
    diametro_pupila = (r_der + r_izq + r_abaj + r_arr)
    
    # Fallback de seguridad
    if diametro_pupila <= 0:
        diametro_pupila = min(h, w) // 4

    return cx, cy, diametro_pupila


def estimar_gaze(cx, cy, ow, oh):
    """
    Estima la dirección de la mirada basándose en la posición relativa
    del iris en relación al centro de la caja delimitadora del ojo.
    
    Retorna:
        dx, dy (desplazamientos de mirada)
        gaze_texto (descripción de la dirección)
    """
    centro_ojo_x = ow / 2.0
    centro_ojo_y = oh / 2.0

    # Desplazamiento respecto al centro
    dx = cx - centro_ojo_x
    dy = cy - centro_ojo_y

    # Umbrales empíricos de desviación (sensibilidad)
    umbral_x = ow * 0.07
    umbral_y = oh * 0.07

    if dx > umbral_x:
        dir_x = "Derecha"
    elif dx < -umbral_x:
        dir_x = "Izquierda"
    else:
        dir_x = "Centro"

    if dy > umbral_y:
        dir_y = "Abajo"
    elif dy < -umbral_y:
        dir_y = "Arriba"
    else:
        dir_y = ""

    gaze_texto = f"{dir_x} {dir_y}".strip()
    return dx, dy, gaze_texto


# ══════════════════════════════════════════════════════════════════════
#                    BUCLE PRINCIPAL DE CAMARA
# ══════════════════════════════════════════════════════════════════════

def iniciar_detector():
    """Abre la cámara y realiza el análisis en tiempo real."""
    ruta_haar_cara = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    ruta_haar_ojos = cv2.data.haarcascades + "haarcascade_eye.xml"

    detector_cara = cv2.CascadeClassifier(ruta_haar_cara)
    detector_ojos = cv2.CascadeClassifier(ruta_haar_ojos)

    if detector_cara.empty() or detector_ojos.empty():
        print("🚨 [ERROR] No se pudieron cargar los archivos Haar Cascades de OpenCV.")
        return

    # Cargar modelo CNN
    modelo, dispositivo, tiene_cnn = cargar_modelo()

    # Abrir la cámara web
    camara = cv2.VideoCapture(0)
    if not camara.isOpened():
        print("🚨 [ERROR] No se pudo abrir la cámara web. Verifica la conexión.")
        return

    print("=" * 60)
    print("  EYESTIM: MVP DE VISIÓN EN TIEMPO REAL")
    print("=" * 60)
    print("  Controles:")
    print("    [Q] o [ESC] -> Salir")
    print("    [S]         -> Guardar captura en carpeta 'capturas/'")
    print("=" * 60)

    capturas_count = 0

    while True:
        ret, frame = camara.read()
        if not ret:
            print("No se pudo leer del flujo de video.")
            break

        # Efecto espejo
        frame = cv2.flip(frame, 1)
        gris_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Detectar caras
        caras = detector_cara.detectMultiScale(
            gris_frame,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(80, 80)
        )

        for (fcx, fcy, fcw, fch) in caras:
            # Dibujar caja del rostro (azul suave)
            cv2.rectangle(frame, (fcx, fcy), (fcx + fcw, fcy + fch), (235, 140, 20), 2)
            cv2.putText(frame, "Rostro", (fcx, fcy - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 140, 20), 2)

            roi_gris = gris_frame[fcy:fcy + fch, fcx:fcx + fcw]
            roi_color = frame[fcy:fcy + fch, fcx:fcx + fcw]

            # 2. Detectar ojos dentro del rostro
            ojos = detector_ojos.detectMultiScale(
                roi_gris,
                scaleFactor=1.1,
                minNeighbors=8,
                minSize=(25, 25)
            )

            for (ox, oy, ow, oh) in ojos:
                recorte_ojo = roi_color[oy:oy + oh, ox:ox + ow]
                if recorte_ojo.size == 0:
                    continue

                # Confirmación IA con la CNN
                es_ojo = True
                certeza = 100.0
                if tiene_cnn:
                    es_ojo, certeza = clasificar_ojo_cnn(recorte_ojo, modelo, dispositivo)

                if es_ojo:
                    # Dibujar caja del ojo (verde vibrante)
                    cv2.rectangle(roi_color, (ox, oy), (ox + ow, oy + oh), (46, 204, 113), 2)

                    # 3. Encontrar centro del iris y tamaño de pupila
                    iris_x, iris_y, diam_pupila = encontrar_centro_y_tamano_iris(recorte_ojo)

                    # Coordenadas globales en el frame completo
                    iris_global_x = fcx + ox + iris_x
                    iris_global_y = fcy + oy + iris_y

                    # 4. Estimar dirección de mirada
                    dx, dy, gaze_texto = estimar_gaze(iris_x, iris_y, ow, oh)

                    # Dibujar pupila estimada (círculo rojo sólido pequeño en el centro)
                    cv2.circle(frame, (iris_global_x, iris_global_y), 3, (0, 0, 255), -1)

                    # Dibujar círculo de la retina/pupila en verde
                    radio_pupila = int(diam_pupila / 2)
                    cv2.circle(frame, (iris_global_x, iris_global_y), radio_pupila, (0, 255, 255), 1)

                    # 5. Dibujar vector de mirada (línea desde el iris hacia afuera)
                    # Multiplicamos la desviación para exagerar visualmente la dirección
                    gaze_vector_x = int(iris_global_x + dx * 5.0)
                    gaze_vector_y = int(iris_global_y + dy * 5.0)
                    cv2.arrowedLine(frame, 
                                    (iris_global_x, iris_global_y), 
                                    (gaze_vector_x, gaze_vector_y), 
                                    (255, 0, 255), 2, tipLength=0.3)

                    # Info en pantalla para cada ojo
                    texto_ojo = f"Mirada: {gaze_texto} | Pupila: {diam_pupila:.1f}px"
                    cv2.putText(frame, texto_ojo, (fcx + ox, fcy + oy - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # Interfaz de usuario (HUD)
        cv2.rectangle(frame, (10, 10), (320, 95), (0, 0, 0), -1) # Panel oscuro de fondo
        cv2.putText(frame, "EYESTIM - MODELO DE VISION MVP", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f"CNN Activa: {'SI' if tiene_cnn else 'NO'}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "[Q] Salir | [S] Guardar Captura", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        # Mostrar video en pantalla
        cv2.imshow("EyeStim MVP - Visualizacion", frame)

        # Atajos de teclado
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q') or tecla == 27:
            break
        elif tecla == ord('s'):
            capturas_count += 1
            dir_capturas = os.path.join(BASE_DIR, "capturas")
            os.makedirs(dir_capturas, exist_ok=True)
            ruta_cap = os.path.join(dir_capturas, f"captura_{capturas_count}.png")
            cv2.imwrite(ruta_cap, frame)
            print(f"📸 Captura guardada en: {ruta_cap}")

    camara.release()
    cv2.destroyAllWindows()
    print("Cámara cerrada correctamente.")

if __name__ == "__main__":
    iniciar_detector()
