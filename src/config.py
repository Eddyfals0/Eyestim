import os

# Rutas del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "bioid")
MODELS_DIR = os.path.join(BASE_DIR, "src", "models")
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "eyestim_cnn.pth")
DOCS_DIR = os.path.join(BASE_DIR, "docs")
LEARNING_CURVE_PATH = os.path.join(DOCS_DIR, "learning_curve.png")

# URLs para la descarga del dataset BioID
BIOID_IMAGES_URL = "https://www.bioid.com/wp-content/uploads/BioID-FaceDatabase-V1.2.zip"
BIOID_POINTS_URL = "https://www.bioid.com/wp-content/uploads/bioidface-db-eyepos.zip"

# Parámetros de procesamiento
ROI_SIZE = 64  # El ojo se recortará a 64x64
JITTER_RANGE = 5  # Rango de jittering para robustez en entrenamiento (píxeles)
RANDOM_SEED = 42

# Parámetros del modelo y entrenamiento
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 15
VAL_SPLIT = 0.2

# Constantes de Pupilometría y Análisis de Atención
PUPIL_ROI_SIZE = 32           # Sub-recorte de pupila centrado en la predicción CNN
CALIBRATION_WINDOW = 100      # Fotogramas para la calibración de la línea base pupilar
GAZE_WINDOW = 30              # Ventana móvil de fotogramas para varianza
GAZE_STABILITY_THRESHOLD = 0.25 # Desviación estándar espacial de distracción
GAZE_CENTER_THRESHOLD = 0.35  # Distancia máxima del centro para clasificar como 'Centro'

