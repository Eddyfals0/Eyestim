import os
import time
import sys
import cv2
import numpy as np
import torch

import config
from model import EyePupilCNN
from pupilometry import estimate_pupil_diameter
from attention import AttentionTracker
from reporter import SessionReporter
from diameter_estimator import LearnedDiameterEstimator


def show_training_dashboard() -> None:
    dashboard_path = os.path.join(config.TRAINING_DOCS_DIR, "project_dashboard.png")
    if not os.path.exists(dashboard_path):
        dashboard_path = os.path.join(
            config.TRAINING_DOCS_DIR, "training_dashboard.png"
        )
    dashboard = cv2.imread(dashboard_path)
    if dashboard is None:
        print(f"[Error] No existe el panel de entrenamiento: {dashboard_path}")
        return
    height, width = dashboard.shape[:2]
    scale = min(1280 / width, 820 / height, 1.0)
    if scale < 1.0:
        dashboard = cv2.resize(dashboard, None, fx=scale, fy=scale)
    cv2.imshow("EyeStim - Evidencia de entrenamiento", dashboard)
    print("Presiona cualquier tecla sobre el panel para cerrarlo.")
    cv2.waitKey(0)
    cv2.destroyWindow("EyeStim - Evidencia de entrenamiento")

def run_realtime_demo():
    print("=== EyeStim: pupilometría e índice experimental de estabilidad ocular ===")
    
    # 1. Configurar dispositivo y cargar modelo CNN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EyePupilCNN().to(device)
    
    if not os.path.exists(config.MODEL_SAVE_PATH):
        print(f"[Error] No se encontró el modelo entrenado en {config.MODEL_SAVE_PATH}")
        print("Por favor, entrena la red primero ejecutando 'python src/train.py'.")
        return
        
    model.load_state_dict(
        torch.load(config.MODEL_SAVE_PATH, map_location=device, weights_only=True)
    )
    model.eval()
    print(f"Modelo cargado. Ejecutando inferencia en {device}.")

    diameter_estimator = None
    if os.path.exists(config.DIAMETER_MODEL_SAVE_PATH):
        diameter_estimator = LearnedDiameterEstimator(
            config.DIAMETER_MODEL_SAVE_PATH, device=device
        )
        print("Modelo EyeDentify de diámetro cargado (salida experimental en mm).")
    else:
        print("[Aviso] No se encontró el modelo EyeDentify de diámetro; se mostrará sólo px.")
    
    # 2. Inicializar analizadores lógicos y de reporte
    tracker = AttentionTracker()
    reporter = SessionReporter()
    
    # 3. Cargar clasificadores Haar Cascades de OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    if face_cascade.empty() or eye_cascade.empty():
        print("[Error] No se pudieron cargar los archivos XML de Haar Cascades de OpenCV.")
        return
        
    # 4. Inicializar captura de video
    cap = cv2.VideoCapture(0)
    has_camera = True
    if not cap.isOpened():
        print("[Advertencia] Cámara web no detectada. Entrando en modo de simulación gráfica de video...")
        has_camera = False
        
    prev_time = time.time()
    frame_count = 0
    
    # Bucle principal de procesamiento
    while True:
        if has_camera:
            ret, frame = cap.read()
            if not ret:
                print("[Error] No se pudo capturar el fotograma.")
                break
        else:
            # Generar fotograma de simulación
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Dibujar cara
            cv2.circle(frame, (320, 240), 100, (60, 60, 60), -1)
            # Dibujar ojos
            cv2.circle(frame, (280, 220), 20, (220, 220, 220), -1)
            cv2.circle(frame, (360, 220), 20, (220, 220, 220), -1)
            cv2.putText(frame, "SIMULACION DE VIDEO", (180, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            time.sleep(0.05)
            
        h_frame, w_frame, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detectar rostros
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))
        
        # Variables por defecto por cuadro si no se detecta nada
        current_diameter = 0.0
        current_diameter_mm = 0.0
        nx, ny = 0.0, 0.0
        attention_score = 0.0
        active_zone = "Centro"
        
        for (x, y, w, h) in faces:
            # Dibujar caja de rostro (Cian elegante)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
            
            # Región facial superior para los ojos
            eye_region_h = int(h * 0.55)
            roi_gray_face = gray[y:y+eye_region_h, x:x+w]
            roi_color_face = frame[y:y+eye_region_h, x:x+w]
            
            # Detectar ojos
            eyes = eye_cascade.detectMultiScale(roi_gray_face, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))
            
            # Procesar hasta 2 ojos
            for idx, (ex, ey, ew, eh) in enumerate(eyes[:2]):
                # Dibujar rectángulo del ojo en verde oscuro
                cv2.rectangle(roi_color_face, (ex, ey), (ex+ew, ey+eh), (0, 150, 0), 1)
                
                # Extraer ROI en gris del ojo
                eye_crop = roi_gray_face[ey:ey+eh, ex:ex+ew]
                
                # Redimensionar y normalizar contraste
                eye_crop_resized = cv2.resize(eye_crop, (config.ROI_SIZE, config.ROI_SIZE))
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                eye_crop_preprocessed = clahe.apply(eye_crop_resized)
                
                # Convertir a tensor [1, 1, 64, 64]
                eye_tensor = torch.tensor(eye_crop_preprocessed.astype(np.float32) / 255.0, dtype=torch.float32)
                eye_tensor = eye_tensor.unsqueeze(0).unsqueeze(0).to(device)
                
                # 1. Inferencia CNN (Centro aproximado de la pupila en [-1, 1])
                with torch.no_grad():
                    pred_t = model(eye_tensor).cpu().squeeze(0)
                
                nx_ojo, ny_ojo = pred_t[0].item(), pred_t[1].item()
                
                # 2. Pupilometría Clásica sobre la ROI
                metrics = estimate_pupil_diameter(eye_crop_resized, (nx_ojo, ny_ojo))
                learned_diameter_mm = (
                    diameter_estimator.predict(eye_crop)
                    if diameter_estimator is not None
                    else 0.0
                )
                
                # Dibujo en pantalla si detecta la pupila
                scale_x = ew / config.ROI_SIZE
                scale_y = eh / config.ROI_SIZE
                
                if metrics.detected:
                    cx_roi, cy_roi = metrics.center_px
                    abs_cx = x + ex + int(cx_roi * scale_x)
                    abs_cy = y + ey + int(cy_roi * scale_y)
                    
                    # Dibujar el contorno de la pupila en verde claro
                    radio_px = int((metrics.diameter_px / 2.0) * scale_x)
                    cv2.circle(frame, (abs_cx, abs_cy), max(2, radio_px), (0, 255, 0), 2)
                    
                    # Dibujar el centro estimado (punto rojo)
                    cv2.circle(frame, (abs_cx, abs_cy), 2, (0, 0, 255), -1)
                    
                    # Dibujar vector de dirección de mirada (Amarillo)
                    gaze_dx = int(nx_ojo * 45)
                    gaze_dy = int(ny_ojo * 45)
                    cv2.line(frame, (abs_cx, abs_cy), (abs_cx + gaze_dx, abs_cy + gaze_dy), (0, 255, 255), 2)
                    
                    # Guardar variables del primer ojo para el tracker general
                    if idx == 0:
                        current_diameter = metrics.diameter_px
                        current_diameter_mm = learned_diameter_mm
                        nx, ny = nx_ojo, ny_ojo
                else:
                    # Si no se detecta la pupila por binarización, usar coordenadas CNN como fallback
                    if idx == 0:
                        current_diameter_mm = learned_diameter_mm
                        nx, ny = nx_ojo, ny_ojo
        
        # 5. Calcular el índice heurístico y mantener la calibración pupilar
        tracker.update_calibration(current_diameter)
        attention_score = tracker.calculate_attention(current_diameter, nx, ny)
        active_zone = tracker.get_active_zone(nx, ny)
        
        # 6. Registrar en el SessionReporter
        reporter.log_frame(
            current_diameter,
            nx,
            ny,
            attention_score,
            active_zone,
            diameter_mm=current_diameter_mm,
        )
        
        # 7. Renderizar HUD Visual Premium en Pantalla
        # Barra horizontal del índice ocular experimental
        bar_x1, bar_y1 = 15, h_frame - 35
        bar_w, bar_h = 160, 15
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + bar_w, bar_y1 + bar_h), (80, 80, 80), -1)
        
        # Color dinámico del índice (no representa probabilidad de atención)
        if attention_score >= 70.0:
            bar_color = (0, 255, 0)      # Verde
        elif attention_score >= 35.0:
            bar_color = (0, 255, 255)    # Amarillo
        else:
            bar_color = (0, 0, 255)      # Rojo
            
        fill_w = int((attention_score / 100.0) * bar_w)
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y1 + bar_h), bar_color, -1)
            
        # Textos de HUD
        cv2.putText(frame, f"Indice ocular*: {attention_score:.0f}/100", (bar_x1 + bar_w + 10, bar_y1 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Estado de calibración
        if not tracker.is_calibrated:
            calib_text = f"Calibrando: {len(tracker.diameter_history)}%"
            cv2.putText(frame, calib_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            cv2.putText(frame, f"Base Pupila: {tracker.baseline_diameter:.1f} px", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        # Diámetro actual y zona de enfoque
        dia_text = f"Pupila: {current_diameter:.1f} px" if current_diameter > 0.0 else "Pupila: --"
        cv2.putText(frame, dia_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Enfoque: {active_zone}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        mm_text = (
            f"Pupila CNN: {current_diameter_mm:.2f} mm*"
            if current_diameter_mm > 0.0
            else "Pupila CNN: --"
        )
        cv2.putText(frame, mm_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 170, 255), 1)
        cv2.putText(frame, "* estimacion experimental, no clinica", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
        
        # FPS en pantalla
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "q: salir/reporte | t: entrenamiento", (w_frame - 250, h_frame - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Mostrar ventana
        cv2.imshow("EyeStim - Pupilometria experimental", frame)
        
        # Salida del visor
        pressed_key = cv2.waitKey(1) & 0xFF
        if pressed_key == ord('q'):
            break
        if pressed_key == ord('t'):
            show_training_dashboard()
            
        frame_count += 1
        if not has_camera and frame_count >= 30:
            print("[Simulación] Finalizado ciclo de prueba de 30 fotogramas.")
            break
            
    # 8. Liberar recursos y generar el reporte analítico al salir
    if has_camera:
        cap.release()
    cv2.destroyAllWindows()
    
    print("\n[Sistema] Sesión finalizada. Generando reportes...")
    reporter.generate_report(tracker.baseline_diameter)
    print("Demo finalizada.")

if __name__ == "__main__":
    if '--training-dashboard' in sys.argv:
        show_training_dashboard()
        raise SystemExit(0)
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        print("[Modo Test] Ejecutando simulación rápida sin cámara.")
        # Generar modelo de prueba dummy si no existe
        if not os.path.exists(config.MODEL_SAVE_PATH):
            os.makedirs(config.MODELS_DIR, exist_ok=True)
            from model import EyePupilCNN
            dummy_model = EyePupilCNN()
            torch.save(dummy_model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"Modelo dummy temporal guardado en {config.MODEL_SAVE_PATH}")
            
    run_realtime_demo()
