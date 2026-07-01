import os
import zipfile
import urllib.request
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import config

def download_bioid(data_dir=config.DATA_DIR):
    """
    Descarga programáticamente el dataset BioID (imágenes y puntos oculares)
    y lo extrae en data_dir si no está ya presente.
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # Comprobar si ya existen archivos pgm y eye
    pgm_files = [f for f in os.listdir(data_dir) if f.endswith(".pgm")]
    eye_files = [f for f in os.listdir(data_dir) if f.endswith(".eye")]
    
    if len(pgm_files) >= 1520 and len(eye_files) >= 1520:
        print(f"[BioID] Dataset ya presente en {data_dir} ({len(pgm_files)} imágenes). Omitiendo descarga.")
        return

    print("[BioID] Descargando dataset BioID (esto puede demorar unos minutos)...")
    
    # Archivos a descargar
    downloads = [
        ("bioid_images.zip", config.BIOID_IMAGES_URL),
        ("bioid_points.zip", config.BIOID_POINTS_URL)
    ]
    
    for filename, url in downloads:
        zip_path = os.path.join(data_dir, filename)
        if not os.path.exists(zip_path):
            print(f"[BioID] Descargando {filename} desde {url}...")
            try:
                # Descarga simple con reporte de progreso
                def progress_hook(count, block_size, total_size):
                    percent = int(count * block_size * 100 / total_size)
                    print(f"\rDescargando: {percent}%", end="")
                
                urllib.request.urlretrieve(url, zip_path, reporthook=progress_hook)
                print(f"\n[BioID] {filename} descargado con éxito.")
            except Exception as e:
                print(f"\n[BioID] Error al descargar {filename}: {e}")
                print("[BioID] Por favor, asegúrate de tener conexión a Internet o descarga los archivos manualmente y colócalos en la carpeta data/bioid/.")
                return

        # Descomprimir
        print(f"[BioID] Extrayendo {filename}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            print(f"[BioID] Extracción de {filename} completa.")
            # Borrar el ZIP para ahorrar espacio
            os.remove(zip_path)
        except Exception as e:
            print(f"[BioID] Error al extraer {filename}: {e}")
            return

    print(f"[BioID] Dataset listo y configurado en {data_dir}.")

class EyeStimDataset(Dataset):
    """
    Cargador de datos personalizado para extraer la ROI del ojo a partir de BioID.
    Mapea cada imagen de BioID en dos muestras: ojo izquierdo y ojo derecho.
    """
    def __init__(self, data_dir=config.DATA_DIR, train=True, jitter_range=config.JITTER_RANGE, roi_size=config.ROI_SIZE):
        self.data_dir = data_dir
        self.train = train
        self.jitter_range = jitter_range
        self.roi_size = roi_size
        self.samples = []
        
        # Escanear el directorio para encontrar pares de archivos .pgm y .eye
        if not os.path.exists(data_dir):
            return
            
        files = sorted(os.listdir(data_dir))
        pgm_basenames = [os.path.splitext(f)[0] for f in files if f.endswith(".pgm")]
        
        for base in pgm_basenames:
            pgm_path = os.path.join(data_dir, f"{base}.pgm")
            eye_path = os.path.join(data_dir, f"{base}.eye")
            
            if os.path.exists(eye_path):
                self.samples.append((pgm_path, eye_path))
                
        if len(self.samples) == 0:
            print(f"[Dataset] Advertencia: No se encontraron muestras en {data_dir}.")

    def __len__(self):
        # Cada imagen tiene 2 ojos
        return len(self.samples) * 2

    def __getitem__(self, idx):
        # Mapear idx a imagen y a qué ojo pertenece (0: izquierdo, 1: derecho)
        sample_idx = idx // 2
        eye_type = idx % 2  # 0 para ojo izquierdo, 1 para derecho
        
        pgm_path, eye_path = self.samples[sample_idx]
        
        # 1. Leer imagen en escala de grises
        img = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen: {pgm_path}")
            
        # 2. Leer coordenadas de los ojos en el archivo .eye
        # Formato esperado: LX LY RX RY (primera línea no comentada)
        with open(eye_path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]
            
        if not lines:
            raise ValueError(f"Archivo de anotaciones vacío o corrupto: {eye_path}")
            
        parts = lines[0].split()
        if len(parts) < 4:
            raise ValueError(f"Formato incorrecto en anotaciones: {eye_path}")
            
        # Coordenadas reales anotadas
        lx, ly = int(parts[0]), int(parts[1])
        rx, ry = int(parts[2]), int(parts[3])
        
        # Seleccionar el ojo correspondiente
        if eye_type == 0:
            cx_real, cy_real = lx, ly
        else:
            cx_real, cy_real = rx, ry
            
        # 3. Aplicar jittering (ruido de desplazamiento) si es entrenamiento
        dx, dy = 0, 0
        if self.train and self.jitter_range > 0:
            dx = np.random.randint(-self.jitter_range, self.jitter_range + 1)
            dy = np.random.randint(-self.jitter_range, self.jitter_range + 1)
            
        # Centro de recorte desplazado por el jitter
        cx_recorte = cx_real + dx
        cy_recorte = cy_real + dy
        
        # 4. Extraer ROI de roi_size x roi_size con padding si excede límites
        x_start = cx_recorte - self.roi_size // 2
        y_start = cy_recorte - self.roi_size // 2
        x_end = x_start + self.roi_size
        y_end = y_start + self.roi_size
        
        h, w = img.shape
        
        # Calcular paddings si nos salimos
        pad_top = max(0, -y_start)
        pad_bottom = max(0, y_end - h)
        pad_left = max(0, -x_start)
        pad_right = max(0, x_end - w)
        
        # Coordenadas de recorte dentro de la imagen
        y1 = max(0, y_start)
        y2 = min(h, y_end)
        x1 = max(0, x_start)
        x2 = min(w, x_end)
        
        crop = img[y1:y2, x1:x2]
        
        # Aplicar padding si es necesario
        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
            
        # 5. Aplicar ecualización local adaptativa CLAHE para contrastar pupila
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        crop = clahe.apply(crop)
        
        # 6. Calcular coordenadas reales de la pupila relativas a la ROI y normalizar a [-1.0, 1.0]
        # La pupila real respecto al centro del recorte está desplazada por -dx, -dy
        px_roi = self.roi_size // 2 - dx
        py_roi = self.roi_size // 2 - dy
        
        # Normalizar al rango [-1.0, 1.0]
        nx = (px_roi - self.roi_size / 2) / (self.roi_size / 2)
        ny = (py_roi - self.roi_size / 2) / (self.roi_size / 2)
        
        # 7. Convertir la imagen a float32 normalizado en [0.0, 1.0]
        crop_norm = crop.astype(np.float32) / 255.0
        
        # Formato de tensores PyTorch [1, H, W] para la imagen y [2] para coordenadas
        image_tensor = torch.tensor(crop_norm, dtype=torch.float32).unsqueeze(0)
        coord_tensor = torch.tensor([nx, ny], dtype=torch.float32)
        
        return image_tensor, coord_tensor

if __name__ == "__main__":
    # Prueba del módulo
    download_bioid()
    
    dataset = EyeStimDataset(train=True)
    print(f"Total de muestras de ojos cargadas: {len(dataset)}")
    
    if len(dataset) > 0:
        img_t, coord_t = dataset[0]
        print(f"Forma del tensor de imagen: {img_t.shape}")
        print(f"Forma del tensor de coordenadas: {coord_t.shape}")
        print(f"Coordenadas normalizadas [x, y]: {coord_t.tolist()}")
        print("Módulo de datos verificado con éxito.")
    else:
        print("Error: No se pudieron cargar muestras.")
