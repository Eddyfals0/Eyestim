import os
import glob
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import math
from model import DetectorDeOjosCNN
from analyze_iris import find_iris_center

def test_on_folder(folder_path="../data/testing"):
    print(f"Buscando imágenes en: {folder_path}")
    
    # 1. Obtener todas las imágenes de la carpeta
    search_pattern = os.path.join(folder_path, "*.*")
    all_files = glob.glob(search_pattern)
    image_paths = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_paths:
        print(f"[X] No se encontraron imagenes en {folder_path}. Por favor pon algunas imagenes de prueba ahi.")
        return
        
    print(f"[OK] Se encontraron {len(image_paths)} imagenes para probar.")

    # 2. Configurar la transformación EXACTA que usamos en entrenamiento (1 canal / Grayscale)
    transformacion = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    to_tensor_simple = transforms.ToTensor()

    # 3. Cargar el modelo entrenado
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = DetectorDeOjosCNN()
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta_modelo = os.path.join(base_dir, "models", "mi_modelo_detectar_ojos.pth")
    
    try:
        modelo.load_state_dict(torch.load(ruta_modelo, map_location=dispositivo))
        print("Modelo Neuronal cargado exitosamente.")
    except FileNotFoundError:
        print(f"[ERROR] No se encontro la red neuronal en {ruta_modelo}. Ejecuta 'python src/train.py' primero.")
        return
        
    modelo.to(dispositivo)
    modelo.eval()

    # 4. Preparar el gráfico de matplotlib
    filas = math.ceil(math.sqrt(len(image_paths)))
    columnas = math.ceil(len(image_paths) / filas)
    
    # Limitar a 16 imágenes máximo para que la pantalla no colapse
    max_images = min(16, len(image_paths))
    filas = math.ceil(math.sqrt(max_images))
    columnas = math.ceil(max_images / filas)
    
    fig, axes = plt.subplots(filas, columnas, figsize=(15, 15))
    axes = axes.flatten() if max_images > 1 else [axes]
    
    print("Analizando...")
    for i in range(max_images):
        ruta_img = image_paths[i]
        try:
            img_cruda = Image.open(ruta_img).convert('RGB')
        except Exception as e:
            print(f"Error abriendo {ruta_img}: {e}")
            continue
            
        # Evaluar con la Red Neuronal
        img_modelo = transformacion(img_cruda).unsqueeze(0).to(dispositivo)
        
        with torch.no_grad():
            salida = modelo(img_modelo)
            porcentaje = torch.sigmoid(salida).item()
            
        # Interpretación
        # Si porcentaje < 0.5 es OJO (Clase 0), si es > 0.5 es SIN_OJO (Clase 1)
        if porcentaje < 0.5:
            certeza = (1 - porcentaje) * 100
            titulo = f"Ojo: {certeza:.1f}%"
            color_titulo = "green"
            # Si es ojo, buscar el centro del iris
            img_tensor = to_tensor_simple(img_cruda)
            x_iris, y_iris = find_iris_center(img_tensor)
        else:
            certeza = porcentaje * 100
            titulo = f"No Ojo: {certeza:.1f}%"
            color_titulo = "red"
            x_iris, y_iris = None, None

        # Graficar
        ax = axes[i]
        ax.imshow(img_cruda)
        
        if x_iris is not None and y_iris is not None:
            ax.scatter(x_iris, y_iris, color='red', s=60, marker='+', linewidth=2)
            
        ax.set_title(titulo, color=color_titulo, fontsize=12)
        ax.axis("off")

    # Apagar los ejes sobrantes en la grilla
    for j in range(max_images, len(axes)):
        axes[j].axis("off")
        
    plt.suptitle("Prueba con Imágenes de Testing en Cualquier Formato", fontsize=20, y=0.98)
    plt.tight_layout()
    
    salida_png = os.path.join(base_dir, "results", "resultados_testing_imagenes.png")
    os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)
    plt.savefig(salida_png)
    print(f"Resultado guardado como imagen en: {salida_png}")
    
    plt.show()

if __name__ == "__main__":
    # La carpeta testing suele estar en data/testing
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    carpeta_test = os.path.join(base_dir, "..", "data", "testing")
    
    # Si no existe la de arriba, buscar dentro del proyecto mismo
    if not os.path.exists(carpeta_test):
        carpeta_test = os.path.join(base_dir, "data", "testing")
        
    # Crear la carpeta si no existe
    if not os.path.exists(carpeta_test):
        os.makedirs(carpeta_test, exist_ok=True)
        print(f"Se ha creado la carpeta {carpeta_test}.")
        print("Por favor añade imágenes ahí para probar la IA y vuelve a ejecutar este script.")
    else:
        test_on_folder(carpeta_test)
