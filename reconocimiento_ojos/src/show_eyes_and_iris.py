import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import math

# Importar funciones de nuestro otro script
from analyze_iris import find_iris_center, get_images
from model import DetectorDeOjosCNN

def show_eyes_and_iris(num_images=16):
    """
    Carga imágenes, usa la red neuronal ya entrenada para confirmar que son ojos,
    y luego grafica la localización del iris.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Intentar buscar el modelo
    ruta_modelo = os.path.join(base_dir, "models", "mi_modelo_detectar_ojos.pth")
    # Respaldos relativos
    if not os.path.exists(ruta_modelo):
        ruta_modelo = os.path.join(base_dir, "..", "models", "mi_modelo_detectar_ojos.pth")
        
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = DetectorDeOjosCNN()
    
    modelo_cargado = False
    try:
        if os.path.exists(ruta_modelo):
            modelo.load_state_dict(torch.load(ruta_modelo, map_location=dispositivo))
            modelo_cargado = True
            print("🧠 Modelo Neuronal cargado exitosamente.")
        else:
            print("🚨 ATENCIÓN: No se encontró la red neuronal.")
            print("Ejecuta 'python src/train.py' primero si quieres que evalúe si es un ojo o no.")
            print("De todas formas, graficaremos el iris asumiendo que las fotos son ojos.")
    except Exception as e:
        print(f"🚨 Hubo un problema cargando la red neuronal: {e}")
        
    if modelo_cargado:    
        modelo.to(dispositivo)
        modelo.eval()

    # Rutas de datos
    ruta_datos = os.path.join(base_dir, "data", "train", "forward_look")
    if not os.path.exists(ruta_datos):
        ruta_datos = os.path.join(base_dir, "..", "data", "train", "forward_look")
    
    # Obtener imágenes
    image_paths = get_images(ruta_datos, num_images)
    if not image_paths:
        return
        
    # Transformaciones para el Modelo Y para Tensor base
    transformacion_modelo = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    to_tensor_simple = transforms.ToTensor()

    # Preparar el Gráfico de matplotlib (ej. 4x4 o 3x3 dependiendo de num_images)
    filas = math.ceil(math.sqrt(len(image_paths)))
    columnas = math.ceil(len(image_paths) / filas)
    
    fig, axes = plt.subplots(filas, columnas, figsize=(15, 15))
    axes = axes.flatten() if len(image_paths) > 1 else [axes]
    
    for i, img_path in enumerate(image_paths):
        img_cruda = Image.open(img_path).convert('RGB')
        
        # 1. EVALUAR CON LA RED NEURONAL (Si está disponible)
        titulo = ""
        color_titulo = "black"
        if modelo_cargado:
            img_modelo = transformacion_modelo(img_cruda).unsqueeze(0).to(dispositivo)
            with torch.no_grad():
                salida = modelo(img_modelo)
                porcentaje = torch.sigmoid(salida).item()
                
            if porcentaje < 0.5:
                certeza = (1 - porcentaje) * 100
                titulo = f"👁️ Ojo: {certeza:.1f}%"
                color_titulo = "green"
            else:
                certeza = porcentaje * 100
                titulo = f"❌ No Ojo: {certeza:.1f}%"
                color_titulo = "red"
        else:
            titulo = f"Imagen {i+1}"
            
        # 2. LOCALIZAR EL IRIS
        img_tensor = to_tensor_simple(img_cruda)
        x_iris, y_iris = find_iris_center(img_tensor)
        
        # 3. GRAFICAR
        ax = axes[i]
        ax.imshow(img_cruda)
        ax.scatter(x_iris, y_iris, color='red', s=60, marker='+', linewidth=2)
        ax.set_title(titulo, color=color_titulo, fontsize=12)
        ax.axis("off")
        
    # Apagar ejes sobrantes
    for j in range(len(image_paths), len(axes)):
        axes[j].axis("off")
        
    plt.suptitle("Red Neuronal + Localizador de Iris", fontsize=20, y=0.98)
    plt.tight_layout()
    
    salida_png = os.path.join(base_dir, "red_neuronal_y_localizacion.png")
    plt.savefig(salida_png)
    print(f"👉 Resultado guardado en: {salida_png}")
    plt.show()

if __name__ == "__main__":
    # Mostrar una matriz de 16 ojos
    print("Iniciando análisis masivo de la red neuronal y del localizador de Iris...")
    show_eyes_and_iris(num_images=16)
