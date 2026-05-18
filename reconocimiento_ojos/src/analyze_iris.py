import os
import glob
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
import matplotlib.pyplot as plt

def get_images(data_dir, num_images=5):
    """Obtiene una lista de rutas de imágenes de la carpeta especificada."""
    # Buscar imágenes .jpg, .png o .jpeg
    search_pattern = os.path.join(data_dir, "*.*")
    all_files = glob.glob(search_pattern)
    image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No se encontraron imágenes en {data_dir}.")
        return []
    
    # Tomar algunas imágenes (hasta num_images)
    return image_files[:num_images]

def find_iris_center(image_tensor):
    """
    Encuentra el centro del iris asumiendo que es la parte más oscura de la imagen.
    Retorna las coordenadas (x, y).
    """
    # Convertir a escala de grises si tiene 3 canales
    if image_tensor.shape[0] == 3:
        # Pesos estándar para convertir RGB a escala de grises
        grayscale = 0.2989 * image_tensor[0] + 0.5870 * image_tensor[1] + 0.1140 * image_tensor[2]
        grayscale = grayscale.unsqueeze(0) # [1, H, W] para que el gaussian_blur funcione
    else:
        grayscale = image_tensor
        
    _, h, w = grayscale.shape
    
    # 1. Tamaño de Kernel Dinámico (10% de la imagen)
    k_size = int(min(h, w) * 0.1)
    if k_size % 2 == 0:
        k_size += 1 # Debe ser impar
    k_size = max(3, k_size)
    sigma = k_size / 3.0
    
    # Aplicar difuminado gaussiano proporcional al tamaño de la imagen
    blurred = F.gaussian_blur(grayscale, kernel_size=[k_size, k_size], sigma=[sigma, sigma])
    
    # 2. Ignorar bordes oscuros (recortar el 15% de los márgenes)
    margin_y = int(h * 0.15)
    margin_x = int(w * 0.15)
    
    center_region = blurred[:, margin_y:h-margin_y, margin_x:w-margin_x]
    
    if center_region.numel() == 0:
        # Fallback si la imagen es excesivamente pequeña
        center_region = blurred
        margin_y, margin_x = 0, 0
    
    # Encontrar el índice del valor mínimo en la región central
    min_idx = torch.argmin(center_region)
    
    # Desenrollar el índice plano en coordenadas 2D (y, x)
    y_center = min_idx // center_region.shape[2]
    x_center = min_idx % center_region.shape[2]
    
    # Sumar el margen recortado para obtener las coordenadas reales
    y = y_center + margin_y
    x = x_center + margin_x
    
    return x.item(), y.item()

def analyze_and_plot(data_dir=None, num_images=5):
    """Lee las imágenes, encuentra el centro del iris y usa matplotlib para mostrar el resultado."""
    # Resolver la ruta de forma segura para que funcione no importa desde dónde se ejecute
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if data_dir is None:
        # Intentar la ruta de imágenes reales primero: Proyecto/data/train/forward_look
        real_data_dir = os.path.join(base_dir, "..", "data", "train", "forward_look")
        
        if not os.path.exists(real_data_dir):
            # Fallback 1: dentro de reconocimiento_ojos/data
            real_data_dir = os.path.join(base_dir, "data", "train", "forward_look")
            if not os.path.exists(real_data_dir):
                # Fallback 2: Las imágenes sintéticas antiguas
                real_data_dir = os.path.join(base_dir, "..", "data", "train", "ojo")
    else:
        real_data_dir = data_dir
        
    print(f"Buscando imágenes en: {real_data_dir}")
    
    image_paths = get_images(real_data_dir, num_images)
    
    if not image_paths:
        return
        
    # Solo transformamos a Tensor para el análisis numérico
    to_tensor = transforms.ToTensor()
    
    # Configurar matplotlib para mostrar las imágenes (1 fila con varias columnas)
    fig, axes = plt.subplots(1, len(image_paths), figsize=(15, 4))
    if len(image_paths) == 1:
        axes = [axes]
        
    for i, img_path in enumerate(image_paths):
        # 1. Cargar imagen original con PIL
        img_pil = Image.open(img_path).convert('RGB')
        
        # 2. Convertir a tensor en PyTorch
        img_tensor = to_tensor(img_pil)
        
        # 3. Encontrar el centro del iris
        center_x, center_y = find_iris_center(img_tensor)
        
        # 4. Graficar
        ax = axes[i]
        ax.imshow(img_pil)
        # Dibujar un punto rojo en el centro detectado
        ax.scatter(center_x, center_y, color='red', s=40, marker='+', linewidth=2)
        ax.set_title(f"Centro: ({center_x}, {center_y})")
        ax.axis('off')
        
    plt.suptitle("Analizador de Localización Central del Iris", fontsize=16)
    plt.tight_layout()
    # Guardar en disco para verificación
    output_path = os.path.join(base_dir, "iris_localization_result.png")
    plt.savefig(output_path)
    print(f"Resultado guardado en: {output_path}")
    plt.show()

if __name__ == "__main__":
    analyze_and_plot(num_images=5)
