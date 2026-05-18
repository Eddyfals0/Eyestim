import torch
import torchvision.transforms as transforms
from PIL import Image
from model import DetectorDeOjosCNN
import sys

def predecir_imagen(ruta_imagen, ruta_modelo="../models/mi_modelo_detectar_ojos.pth"):
    # 1. Definir cómo transformar la foto a analizar 
    # (DEBE SER LA MISMA TRANSFORMACIÓN exactita que usamos para entrenar)
    transformacion = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # 2. Cargar el "cerebro" (Los pesos que guardamos al terminar train.py)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = DetectorDeOjosCNN()
    
    try:
        # Cargar archivo guardado en el modelo vacío
        modelo.load_state_dict(torch.load(ruta_modelo, map_location=dispositivo))
    except FileNotFoundError:
        print(f"🚨 ¡Alto! No encontré el archivo del modelo en {ruta_modelo}. ¿Ya ejecutaste train.py?")
        return
        
    modelo.to(dispositivo)
    modelo.eval() # Modo evaluación (no aprende nada ahora, solo aplica lo que ya sabe)

    # 3. Leer y transformar tu imagen personal
    try:
        imagen_cruda = Image.open(ruta_imagen).convert("RGB")
    except FileNotFoundError:
        print(f"🚨 No se pudo encontrar tu imagen: {ruta_imagen}")
        return

    # Aplicamos tamaño y creamos la "dimensión del Lote". (1 foto en vez de un bloque de 32)
    imagen_tensor = transformacion(imagen_cruda).unsqueeze(0).to(dispositivo)

    # 4. Magia de predicción
    with torch.no_grad(): # No necesitamos registrar pasos de error, solo la respuesta
        salida_cruda = modelo(imagen_tensor)
        # Aplicamos la función Sigmoide para que el número del modelo quede de 0.0 a 1.0 (Porcentaje)
        porcentaje = torch.sigmoid(salida_cruda).item()

    # Como las clases se organizaron alfabéticamente (0 = ojo, 1 = sin_ojo),
    # Si porcentaje > 0.5 significa que es clase 1 (sin_ojo), de lo contrario es clase 0 (ojo).
    if porcentaje < 0.5:
        certeza = (1 - porcentaje) * 100
        print(f"👁️ ¡HE DETECTADO UN OJO! (Estoy un {certeza:.1f}% seguro)")
    else:
        certeza = porcentaje * 100
        print(f"❌ Aquí no hay ningún ojo detectado. (Estoy un {certeza:.1f}% seguro)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso sugerido: python predict.py <ruta_de_tu_imagen.jpg>")
    else:
        imagen_a_evaluar = sys.argv[1]
        predecir_imagen(imagen_a_evaluar)
