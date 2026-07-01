import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

import config
from dataset import EyeStimDataset
from model import EyePupilCNN

def evaluate_model():
    print("=== Evaluando Modelo de Estimación Estática ===")
    
    # 1. Instanciar dataset en modo evaluación (sin jitter)
    dataset = EyeStimDataset(train=False)
    if len(dataset) == 0:
        print("[Error] No hay muestras disponibles para evaluar.")
        return
        
    print(f"Total de muestras de prueba a evaluar: {len(dataset)}")
    
    # 2. Cargar modelo entrenado
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EyePupilCNN().to(device)
    
    if not os.path.exists(config.MODEL_SAVE_PATH):
        print(f"[Error] Archivo de modelo no encontrado en {config.MODEL_SAVE_PATH}")
        print("Por favor, ejecuta el entrenamiento primero con 'python src/train.py'.")
        return
        
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    model.eval()
    print(f"Pesos cargados correctamente desde {config.MODEL_SAVE_PATH}")
    
    # 3. Calcular errores euclidianos promedio
    errors = []
    visualize_samples = []
    
    with torch.no_grad():
        for idx in range(min(len(dataset), 500)):  # Evaluar hasta 500 muestras para estadísticas
            image_t, target_t = dataset[idx]
            
            # Formato batch [1, 1, 64, 64]
            input_t = image_t.unsqueeze(0).to(device)
            output_t = model(input_t).cpu().squeeze(0)
            
            # Coordenadas reales y predichas en [-1, 1]
            rx_norm, ry_norm = target_t[0].item(), target_t[1].item()
            px_norm, py_norm = output_t[0].item(), output_t[1].item()
            
            # Des-normalizar a píxeles de la ROI de 64x64
            rx_px = (rx_norm + 1.0) * (config.ROI_SIZE / 2.0)
            ry_px = (ry_norm + 1.0) * (config.ROI_SIZE / 2.0)
            px_px = (px_norm + 1.0) * (config.ROI_SIZE / 2.0)
            py_px = (py_norm + 1.0) * (config.ROI_SIZE / 2.0)
            
            # Distancia euclidiana en píxeles
            dist = np.sqrt((rx_px - px_px)**2 + (ry_px - py_px)**2)
            errors.append(dist)
            
            # Guardar algunas muestras para visualización gráfica
            if idx < 6:  # Guardar las primeras 6 para graficarlas
                visualize_samples.append({
                    "image": image_t.squeeze(0).numpy(),
                    "real": (rx_px, ry_px),
                    "pred": (px_px, py_px),
                    "error": dist
                })
                
    mean_error = np.mean(errors)
    print(f"\nResultados de la Evaluación:")
    print(f"  - Error Euclidiano Promedio: {mean_error:.4f} píxeles (en ROI {config.ROI_SIZE}x{config.ROI_SIZE})")
    
    # 4. Graficar y guardar panel de predicciones de muestra
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(f"Ejemplos de Estimación de Ojos (Error Medio: {mean_error:.2f} px)", fontsize=16)
    
    for i, sample in enumerate(visualize_samples):
        ax = axes[i // 3, i % 3]
        ax.imshow(sample["image"], cmap="gray")
        
        # Ojo real (punto verde)
        rx, ry = sample["real"]
        ax.plot(rx, ry, 'go', markersize=8, label="Real" if i == 0 else "")
        
        # Ojo predicho (punto rojo)
        px, py = sample["pred"]
        ax.plot(px, py, 'ro', markersize=8, label="Pred" if i == 0 else "")
        
        ax.set_title(f"Muestra {i+1} | Err: {sample['error']:.2f} px")
        ax.axis("off")
        if i == 0:
            ax.legend(loc="upper left")
            
    plt.tight_layout()
    output_vis_path = os.path.join(config.DOCS_DIR, "sample_predictions.png")
    plt.savefig(output_vis_path)
    plt.close()
    print(f"Panel visual de predicciones guardado en: {output_vis_path}")

if __name__ == "__main__":
    evaluate_model()
