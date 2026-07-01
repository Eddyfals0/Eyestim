import cv2
import numpy as np
import torch
import config

class PupilMetrics:
    """
    Estructura de datos para almacenar el resultado físico de la estimación de la pupila.
    """
    def __init__(self, diameter: float, area: float, center: tuple[float, float], detected: bool):
        self.diameter_px = diameter
        self.area_px = area
        self.center_px = center
        self.detected = detected

    def to_dict(self):
        return {
            "diameter_px": self.diameter_px,
            "area_px": self.area_px,
            "center_px": self.center_px,
            "detected": self.detected
        }

def estimate_pupil_diameter(eye_roi: np.ndarray, predicted_center_norm: tuple[float, float], roi_size: int = config.ROI_SIZE) -> PupilMetrics:
    """
    Aplica procesamiento clásica (umbralización adaptativa local y ajuste de elipse)
    sobre la ROI del ojo centrado en la predicción previa de la CNN.
    """
    # 1. Desnormalizar las coordenadas CNN en [-1, 1] a píxeles absolutos de la ROI de 64x64
    nx, ny = predicted_center_norm
    cx_px = int((nx + 1.0) * (roi_size / 2.0))
    cy_px = int((ny + 1.0) * (roi_size / 2.0))
    
    # 2. Extraer sub-recorte centrado en la predicción CNN
    ps = config.PUPIL_ROI_SIZE
    x1 = max(0, cx_px - ps // 2)
    y1 = max(0, cy_px - ps // 2)
    x2 = min(roi_size, cx_px + ps // 2)
    y2 = min(roi_size, cy_px + ps // 2)
    
    pupil_crop = eye_roi[y1:y2, x1:x2]
    if pupil_crop.size == 0 or pupil_crop.shape[0] < 5 or pupil_crop.shape[1] < 5:
        return PupilMetrics(0.0, 0.0, (0.0, 0.0), False)
        
    # 3. Aplicar filtrado Gaussiano para reducir ruido de pestañas o cejas
    blur = cv2.GaussianBlur(pupil_crop, (5, 5), 0)
    
    # 4. Encontrar intensidad mínima (la pupila es el área más oscura de la ROI)
    min_val, _, _, _ = cv2.minMaxLoc(blur)
    
    # 5. Umbral local dinámico (binarización) centrado en el color de la pupila
    threshold_value = min(255, max(0, int(min_val + 15)))
    _, thresh = cv2.threshold(blur, threshold_value, 255, cv2.THRESH_BINARY_INV)
    
    # 6. Limpieza morfológica para rellenar huecos
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # 7. Buscar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_contour = None
    max_area = 0.0
    
    for c in contours:
        area = cv2.contourArea(c)
        if area > max_area:
            max_area = area
            best_contour = c
            
    # 8. Ajustar elipse si el contorno es lo suficientemente grande y cerrado
    if best_contour is not None and max_area > 8 and len(best_contour) >= 5:
        try:
            ellipse = cv2.fitEllipse(best_contour)
            (ecx, ecy), (major_axis, minor_axis), angle = ellipse
            
            # Re-mapear las coordenadas locales de la elipse de vuelta a la ROI de 64x64
            abs_cx = x1 + ecx
            abs_cy = y1 + ecy
            
            diameter = (major_axis + minor_axis) / 2.0
            
            # Restricciones razonables de tamaño físico de pupila en una ROI de 64
            if 2.0 <= diameter <= 25.0:
                return PupilMetrics(diameter, float(max_area), (abs_cx, abs_cy), True)
        except Exception:
            pass
            
    return PupilMetrics(0.0, 0.0, (0.0, 0.0), False)

if __name__ == "__main__":
    # Test unitario básico con un ojo sintético (círculo oscuro)
    print("=== Test Unitario Autónomo de Pupilometría ===")
    
    # Crear ROI de ojo simulada de 64x64 en gris claro
    sim_eye = np.ones((64, 64), dtype=np.uint8) * 180
    # Dibujar pupila oscura de radio 6 (diámetro 12) centrada en (32, 32)
    cv2.circle(sim_eye, (32, 32), 6, (30), -1)
    # Suavizado para simular desenfoque biológico
    sim_eye = cv2.GaussianBlur(sim_eye, (3, 3), 0)
    
    # Simular una predicción CNN con un ligero desplazamiento en [-1, 1]
    # En (32, 32) nx=0.0, ny=0.0. Probemos con una predicción perfecta (0.0, 0.0)
    metrics = estimate_pupil_diameter(sim_eye, (0.0, 0.0))
    
    print(f"Pupila detectada: {metrics.detected}")
    print(f"Diámetro estimado: {metrics.diameter_px:.2f} px (Real: ~12.0 px)")
    print(f"Centro estimado: {metrics.center_px} (Real: (32.0, 32.0))")
    print(f"Área estimada: {metrics.area_px} px^2")
    
    assert metrics.detected, "Error: Debería haber detectado la pupila."
    assert 7.0 <= metrics.diameter_px <= 13.0, "Error: Estimación del diámetro fuera de rango."
    print("Módulo de pupilometría validado correctamente en aislamiento.")
