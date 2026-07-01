import numpy as np
import config

class AttentionTracker:
    """
    Gestiona la calibración de la línea base pupilar, clasifica espacialmente la mirada 
    a regiones y evalúa el porcentaje dinámico de atención.
    """
    def __init__(self, calibration_window: int = config.CALIBRATION_WINDOW, gaze_window: int = config.GAZE_WINDOW):
        self.calibration_window = calibration_window
        self.gaze_window = gaze_window
        
        # Historial de calibración
        self.diameter_history = []
        self.baseline_diameter = 0.0
        self.is_calibrated = False
        
        # Historial de mirada (ventana móvil)
        self.gaze_history_x = []
        self.gaze_history_y = []

    def get_active_zone(self, nx: float, ny: float) -> str:
        """
        Clasifica las coordenadas normalizadas [-1, 1] de la mirada en una de las 5 regiones.
        """
        magnitude = np.sqrt(nx**2 + ny**2)
        if magnitude < config.GAZE_CENTER_THRESHOLD:
            return "Centro"
            
        # Determinar dirección predominante usando diagonales absolutas
        if ny < -abs(nx):
            return "Arriba"
        elif ny > abs(nx):
            return "Abajo"
        elif nx < -abs(ny):
            return "Izquierda"
        elif nx > abs(ny):
            return "Derecha"
        return "Centro"

    def update_calibration(self, diameter: float) -> None:
        """
        Acumula muestras válidas de diámetros para fijar la línea base.
        """
        if self.is_calibrated or diameter <= 0.0:
            return
            
        self.diameter_history.append(diameter)
        if len(self.diameter_history) >= self.calibration_window:
            # Usar la mediana para evitar que outliers de parpadeo afecten la línea base
            self.baseline_diameter = float(np.median(self.diameter_history))
            self.is_calibrated = True
            print(f"\n[AttentionTracker] Calibración completa. Diámetro base: {self.baseline_diameter:.2f} px")

    def calculate_attention(self, current_diameter: float, nx: float, ny: float) -> float:
        """
        Calcula el score de atención cognitiva (0% - 100%) combinando la
        estabilidad de la mirada y los cambios relativos en el diámetro.
        """
        # Registrar mirada en la ventana móvil
        self.gaze_history_x.append(nx)
        self.gaze_history_y.append(ny)
        
        if len(self.gaze_history_x) > self.gaze_window:
            self.gaze_history_x.pop(0)
            self.gaze_history_y.pop(0)
            
        # 1. Parpadeo u ojo cerrado -> Atención cae a 0 inmediatamente
        if current_diameter <= 0.0:
            return 0.0
            
        # 2. Calcular la estabilidad de la mirada (inversa de la desviación estándar)
        if len(self.gaze_history_x) >= 5:
            std_x = np.std(self.gaze_history_x)
            std_y = np.std(self.gaze_history_y)
            std_total = np.sqrt(std_x**2 + std_y**2)
            
            # Si std_total es 0, estabilidad es 1.0. Si supera el umbral, cae a 0.0
            stability = max(0.0, 1.0 - (std_total / config.GAZE_STABILITY_THRESHOLD))
        else:
            stability = 0.5  # Valor por defecto inicial mientras se llena la ventana móvil
            
        # 3. Evaluar dilatación cognitiva respecto a la línea base
        pupil_factor = 1.0
        if self.is_calibrated and self.baseline_diameter > 0.0:
            dilation_ratio = current_diameter / self.baseline_diameter
            
            # Dilatación cognitiva leve (esfuerzo de atención/procesamiento mental)
            if 1.02 <= dilation_ratio <= 1.20:
                pupil_factor = 1.15
            # Contracción extrema o dilatación por luz/ruido físico
            elif dilation_ratio < 0.85 or dilation_ratio > 1.30:
                pupil_factor = 0.80
                
        # 4. Score final compuesto de atención
        score = stability * pupil_factor * 100.0
        return float(np.clip(score, 0.0, 100.0))

if __name__ == "__main__":
    # Test unitario autónomo
    print("=== Test Unitario Autónomo del Analizador de Atención ===")
    tracker = AttentionTracker(calibration_window=10, gaze_window=5)
    
    # 1. Test de clasificación de zonas
    print(f"Zona (0.0, 0.0): {tracker.get_active_zone(0.0, 0.0)} (Esperado: Centro)")
    print(f"Zona (0.8, 0.0): {tracker.get_active_zone(0.8, 0.0)} (Esperado: Derecha)")
    print(f"Zona (0.0, -0.9): {tracker.get_active_zone(0.0, -0.9)} (Esperado: Arriba)")
    
    assert tracker.get_active_zone(0.0, 0.0) == "Centro"
    assert tracker.get_active_zone(0.8, 0.0) == "Derecha"
    assert tracker.get_active_zone(0.0, -0.9) == "Arriba"
    
    # 2. Test de calibración
    for val in [10.0, 10.2, 9.8, 10.0, 10.1, 9.9, 10.0, 10.2, 9.8, 10.0]:
        tracker.update_calibration(val)
        
    print(f"¿Calibrado?: {tracker.is_calibrated} | Diámetro base: {tracker.baseline_diameter:.2f} px")
    assert tracker.is_calibrated, "Error: Debería haberse calibrado."
    assert abs(tracker.baseline_diameter - 10.0) < 0.5, "Error: Línea base incorrecta."
    
    # 3. Test de atención
    # Caso 1: Mirada fija (estabilidad alta) y pupila normal
    for _ in range(5):
        score_fijo = tracker.calculate_attention(10.0, 0.1, 0.1)
    print(f"Score mirada fija: {score_fijo:.2f}% (Esperado: > 80%)")
    assert score_fijo > 80.0
    
    # Caso 2: Mirada muy errática (desviaciones grandes)
    tracker.calculate_attention(10.0, -0.9, 0.9)
    tracker.calculate_attention(10.0, 0.9, -0.9)
    score_erratico = tracker.calculate_attention(10.0, 0.0, 0.0)
    print(f"Score mirada errática: {score_erratico:.2f}% (Esperado: < 40%)")
    assert score_erratico < 40.0
    
    print("Módulo de análisis de atención verificado correctamente.")
