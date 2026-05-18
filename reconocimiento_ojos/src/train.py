import os
import torch
import torch.nn as nn
import torch.optim as optim
from model import DetectorDeOjosCNN
from dataset import preparar_datos
import csv
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

# Configuración del entrenamiento
EPOCAS = 10                  # Número de veces que el modelo revisará TODAS las imágenes (Vueltas)
TASA_APRENDIZAJE = 0.001     # Qué tan rápidos o bruscos son los ajustes al equivocarse (Learning Rate)
RUTA_DATOS = "../data"       

def entrenar():
    # 1. Preparar la estructura
    os.makedirs("../models", exist_ok=True)
    
    print("1️⃣ Preparando e inspeccionando imágenes...")
    cargador_entrenamiento, cargador_validacion, clases = preparar_datos(RUTA_DATOS)
    print(f"👉 Clases detectadas: {clases}")

    # 2. Inicializar el motor
    # Usa la Tarjeta Gráfica (GPU) si está instalada, si no, usa el CPU normal
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"2️⃣ Cargando modelo en: {dispositivo}")
    modelo = DetectorDeOjosCNN().to(dispositivo)

    # 3. Definir Herramientas de corrección
    # BCE = Binary Cross Entropy. Especializada cuando solo hay 2 opciones (Ojo vs No-Ojo)
    criterio_error = nn.BCEWithLogitsLoss() 
    
    # El Optimizador (Adam) es el 'mecánico' que ajusta lo que el modelo aprendió basándose en el error
    optimizador = optim.Adam(modelo.parameters(), lr=TASA_APRENDIZAJE)

    # 4. Iniciar Bucle de Entrenamiento (La Escuelita)
    print("3️⃣ ¡Iniciando el entrenamiento!")
    
    for epoca in range(EPOCAS):
        modelo.train() # Decirle al modelo que está en modo de ENTRENAMIENTO
        error_acumulado = 0.0
        
        for imagenes, etiquetas_reales in cargador_entrenamiento:
            # Enviar datos a la tarjeta gráfica / cpu
            imagenes = imagenes.to(dispositivo)
            # Acoplar el formato al esperado (matriz de X renglones, 1 columna)
            etiquetas_reales = etiquetas_reales.view(-1, 1).float().to(dispositivo)

            # Paso 1: Resetear el cerebro para no recordar errores pasados
            optimizador.zero_grad() 
            
            # Paso 2: Preguntarle al modelo qué cree que hay en las fotos
            prediccion = modelo(imagenes) 
            
            # Paso 3: Calcular el regaño (Loss)
            error = criterio_error(prediccion, etiquetas_reales) 
            
            # Paso 4: Backpropagation (El modelo se da cuenta de SU culpa matemática en cada capa)
            error.backward() 
            
            # Paso 5: Optimización (El mecánico ajusta las tuercas basado en la culpa)
            optimizador.step() 
            
            error_acumulado += error.item()

        # Al final de la época (la vuelta de estudio), imprimimos cómo le fue:
        error_promedio = error_acumulado / len(cargador_entrenamiento)
        print(f"🔄 Época {epoca+1}/{EPOCAS} | Puntuación de Error: {error_promedio:.4f}")

    # 5. Guardar modelo final para la posteridad
    ruta_guardado = "../models/mi_modelo_detectar_ojos.pth"
    torch.save(modelo.state_dict(), ruta_guardado)
    print(f"✅ ¡Terminamos! Modelo guardado exitosamente en: {ruta_guardado}")

if __name__ == "__main__":
    entrenar()

class DatasetFeedback(Dataset):
    def __init__(self, ruta_csv):
        self.datos = []
        self.transformacion_modelo = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        with open(ruta_csv, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader) # Saltar cabecera
            for row in reader:
                ruta_img, etiqueta_texto = row
                # clase 0 = ojo, clase 1 = sin_ojo (igual que en predict/train)
                clase_num = 0.0 if etiqueta_texto == "ojo" else 1.0
                self.datos.append((ruta_img, clase_num))
                
    def __len__(self):
        return len(self.datos)
        
    def __getitem__(self, idx):
        ruta_img, etiqueta = self.datos[idx]
        img = Image.open(ruta_img).convert('RGB')
        img_tensor = self.transformacion_modelo(img)
        return img_tensor, torch.tensor(etiqueta, dtype=torch.float32)

def reentrenar(ruta_csv):
    """
    Toma el modelo actual y lo ajusta (fine-tuning) usando SOLAMENTE las 
    imágenes documentadas en el CSV de feedback (gui_validator.py).
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta_modelo = os.path.join(base_dir, "models", "mi_modelo_detectar_ojos.pth")
    if not os.path.exists(ruta_modelo):
        print("No hay modelo original para re-entrenar.")
        return False
        
    # 1. Cargar el modelo existente
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = DetectorDeOjosCNN()
    modelo.load_state_dict(torch.load(ruta_modelo, map_location=dispositivo))
    modelo.to(dispositivo)
    
    # 2. Preparar los datos nuevos
    try:
        dataset_nuevo = DatasetFeedback(ruta_csv)
        if len(dataset_nuevo) == 0:
            print("El CSV está vacío.")
            return False
            
        cargador = DataLoader(dataset_nuevo, batch_size=4, shuffle=True)
    except Exception as e:
        print(f"Error cargando feedback: {e}")
        return False
        
    # 3. Entrenamiento breve (Fine-Tuning)
    criterio_error = nn.BCEWithLogitsLoss() 
    # Tasa de aprendizaje más baja para no olvidar lo que ya sabía (Fine-tuning)
    optimizador = optim.Adam(modelo.parameters(), lr=0.0005) 
    
    EPOCAS_REFUERZO = 5
    modelo.train()
    
    print(f"🔄 Iniciando Re-entrenamiento con {len(dataset_nuevo)} imágenes corregidas...")
    for epoca in range(EPOCAS_REFUERZO):
        error_acumulado = 0.0
        for imagenes, etiquetas_reales in cargador:
            imagenes = imagenes.to(dispositivo)
            etiquetas_reales = etiquetas_reales.view(-1, 1).to(dispositivo)
            
            optimizador.zero_grad() 
            prediccion = modelo(imagenes) 
            error = criterio_error(prediccion, etiquetas_reales) 
            error.backward() 
            optimizador.step() 
            
            error_acumulado += error.item()
            
        print(f"   Refuerzo {epoca+1}/{EPOCAS_REFUERZO} - Error: {error_acumulado/len(cargador):.4f}")
        
    # Guardar modelo reentrenado sobreescribiendo el anterior
    torch.save(modelo.state_dict(), ruta_modelo)
    print("✅ Red Neuronal actualizada exitosamente con el aprendizaje.")
    return True

