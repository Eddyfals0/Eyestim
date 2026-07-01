import sys
import torch
import cv2
import numpy as np

def check_dependencies():
    """
    Verifica que las dependencias clave estén instaladas correctamente y sean accesibles.
    """
    print("=== EyeStim Dependency Check ===")
    
    # 1. Python Version
    print(f"Python Version: {sys.version}")
    
    # 2. PyTorch
    print(f"PyTorch Version: {torch.__version__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch Device: {device} (CUDA available: {torch.cuda.is_available()})")
    
    # 3. OpenCV
    print(f"OpenCV Version: {cv2.__version__}")
    
    # 4. NumPy
    print(f"NumPy Version: {np.__version__}")
    
    # Probar que OpenCV funciona creando una imagen básica y validando sus funciones
    try:
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.putText(dummy_img, "Test", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        _, encoded = cv2.imencode(".png", dummy_img)
        print("OpenCV dummy drawing: OK")
        
        # Test unitario para cv2.fitEllipse (necesitamos al menos 5 puntos)
        points = np.array([[20, 20], [20, 40], [30, 45], [40, 40], [40, 20]], dtype=np.int32)
        ellipse = cv2.fitEllipse(points)
        print("OpenCV fitEllipse: OK")
    except Exception as e:
        print(f"OpenCV test drawing/fitEllipse failed: {e}")
        return False
        
    print("All core libraries are installed and operational.")
    return True

if __name__ == "__main__":
    success = check_dependencies()
    sys.exit(0 if success else 1)
