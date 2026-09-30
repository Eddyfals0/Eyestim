import os
import csv
import zipfile
import urllib.request
from pathlib import Path
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


class EyeDentifyDiameterDataset(Dataset):
    """EyeDentify left-eye crops with subject-disjoint manifest splits."""

    def __init__(
        self,
        manifest_path=config.EYEDENTIFY_MANIFEST_PATH,
        split="train",
        training=False,
        target_mean=0.0,
        target_std=1.0,
    ):
        self.manifest_path = os.path.abspath(manifest_path)
        self.data_root = os.path.dirname(self.manifest_path)
        self.split = split
        self.training = training
        self.target_mean = float(target_mean)
        self.target_std = max(float(target_std), 1e-6)
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(
                f"No se encontró el manifiesto EyeDentify: {self.manifest_path}"
            )
        with open(self.manifest_path, "r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.rows = [
            row
            for row in rows
            if row.get("split") == split
            and row.get("left_pupil_mm")
            and os.path.exists(os.path.join(self.data_root, row["image_path"]))
        ]
        if not self.rows:
            raise ValueError(f"EyeDentify no contiene muestras para split={split!r}")

    def __len__(self):
        return len(self.rows)

    @staticmethod
    def preprocess(image):
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = cv2.resize(
            image,
            (config.DIAMETER_INPUT_WIDTH, config.DIAMETER_INPUT_HEIGHT),
            interpolation=cv2.INTER_CUBIC,
        )
        return image

    def _augment(self, image):
        if np.random.random() < 0.8:
            alpha = np.random.uniform(0.75, 1.25)
            beta = np.random.uniform(-18.0, 18.0)
            image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        if np.random.random() < 0.3:
            image = cv2.GaussianBlur(image, (3, 3), 0)
        if np.random.random() < 0.5:
            image = cv2.flip(image, 1)
        if np.random.random() < 0.35:
            noise = np.random.normal(0.0, 3.0, image.shape)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return image

    def __getitem__(self, idx):
        row = self.rows[idx]
        image_path = os.path.join(self.data_root, row["image_path"])
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"No se pudo leer {image_path}")
        image = self.preprocess(image)
        if self.training:
            image = self._augment(image)
        image_tensor = torch.from_numpy(
            np.transpose(image.astype(np.float32) / 255.0, (2, 0, 1)).copy()
        )
        target_mm = float(row["left_pupil_mm"])
        target_norm = (target_mm - self.target_mean) / self.target_std
        return {
            "image": image_tensor,
            "target": torch.tensor(target_norm, dtype=torch.float32),
            "target_mm": torch.tensor(target_mm, dtype=torch.float32),
            "participant_id": int(row["participant_id"]),
            "session_id": int(row["session_id"]),
            "image_path": image_path,
        }


def prepare_lpw_cache(data_dir=config.LPW_CENTER_DIR, roi_size=config.ROI_SIZE):
    """Decode LPW videos once into compact 64x64 NumPy caches."""
    cache_paths = []
    root = os.path.abspath(data_dir)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for video_path in sorted(Path(root).glob("*/*.avi")):
        labels_path = video_path.with_suffix(".txt")
        cache_path = video_path.with_suffix(f".roi{roi_size}.npz")
        cache_paths.append(cache_path)
        if cache_path.exists() and cache_path.stat().st_size > 0:
            continue
        labels = np.loadtxt(labels_path, dtype=np.float32)
        capture = cv2.VideoCapture(str(video_path))
        images = []
        targets = []
        frame_index = 0
        while frame_index < len(labels):
            ok, frame = capture.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            resized = cv2.resize(
                gray, (roi_size, roi_size), interpolation=cv2.INTER_AREA
            )
            images.append(clahe.apply(resized))
            x, y = labels[frame_index]
            targets.append((2.0 * x / width - 1.0, 2.0 * y / height - 1.0))
            frame_index += 1
        capture.release()
        if not images:
            raise RuntimeError(f"No se decodificaron frames de {video_path}")
        temporary = cache_path.with_suffix(cache_path.suffix + ".part.npz")
        np.savez_compressed(
            temporary,
            images=np.asarray(images, dtype=np.uint8),
            targets=np.asarray(targets, dtype=np.float32),
        )
        os.replace(temporary, cache_path)
    return cache_paths


class LPWCenterDataset(Dataset):
    """Pupil-centre regression data from participant-disjoint LPW videos."""

    def __init__(
        self,
        participants,
        data_dir=config.LPW_CENTER_DIR,
        training=False,
    ):
        prepare_lpw_cache(data_dir)
        images = []
        targets = []
        participant_values = []
        frame_values = []
        for participant in participants:
            participant_dir = Path(data_dir) / str(participant)
            caches = sorted(participant_dir.glob(f"*.roi{config.ROI_SIZE}.npz"))
            if not caches:
                raise FileNotFoundError(f"No hay cache LPW para participante {participant}")
            for cache in caches:
                content = np.load(cache)
                sample_count = len(content["images"])
                images.append(content["images"])
                targets.append(content["targets"])
                participant_values.extend([participant] * sample_count)
                frame_values.extend(range(sample_count))
        self.images = np.concatenate(images)
        self.targets = np.concatenate(targets)
        self.participants = np.asarray(participant_values, dtype=np.int64)
        self.frames = np.asarray(frame_values, dtype=np.int64)
        self.training = training

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].copy()
        target = self.targets[idx].copy()
        if self.training:
            if np.random.random() < 0.75:
                alpha = np.random.uniform(0.75, 1.25)
                beta = np.random.uniform(-15.0, 15.0)
                image = np.clip(
                    image.astype(np.float32) * alpha + beta, 0, 255
                ).astype(np.uint8)
            if np.random.random() < 0.5:
                image = cv2.flip(image, 1)
                target[0] *= -1.0
            shift_x = np.random.randint(-4, 5)
            shift_y = np.random.randint(-4, 5)
            transform = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            image = cv2.warpAffine(
                image,
                transform,
                (config.ROI_SIZE, config.ROI_SIZE),
                borderMode=cv2.BORDER_REFLECT_101,
            )
            target += np.asarray(
                [2.0 * shift_x / config.ROI_SIZE, 2.0 * shift_y / config.ROI_SIZE],
                dtype=np.float32,
            )
            target = np.clip(target, -1.0, 1.0)
        return {
            "image": torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0),
            "target": torch.from_numpy(target),
            "participant_id": int(self.participants[idx]),
            "frame": int(self.frames[idx]),
        }

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
