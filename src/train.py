import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

import config
from dataset import download_bioid, EyeStimDataset
from model import EyePupilCNN

def train_model():
    print("=== Iniciando Entrenamiento de EyeStim ===")
    
    # 1. Asegurar descarga de datos
    download_bioid()
    
    # 2. Instanciar dataset y realizar partición entrenamiento/validación
    full_dataset = EyeStimDataset(train=True)
    if len(full_dataset) == 0:
        print("[Error] Dataset vacío. No se puede entrenar.")
        return
        
    val_size = int(len(full_dataset) * config.VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    
    # Semilla fija para reproducibilidad
    torch.manual_seed(config.RANDOM_SEED)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # El de validación no debería llevar jittering
    val_dataset.dataset.train = False
    
    # 3. Crear DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    print(f"Muestras de entrenamiento: {len(train_dataset)}")
    print(f"Muestras de validación: {len(val_dataset)}")
    
    # 4. Inicializar modelo, pérdida y optimizador
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EyePupilCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Listas para almacenar métricas
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.DOCS_DIR, exist_ok=True)
    
    # 5. Bucle de Entrenamiento
    for epoch in range(1, config.EPOCHS + 1):
        # Modo Entrenamiento
        model.train()
        running_train_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * images.size(0)
            
        epoch_train_loss = running_train_loss / len(train_dataset)
        train_losses.append(epoch_train_loss)
        
        # Modo Evaluación (Validación)
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item() * images.size(0)
                
        epoch_val_loss = running_val_loss / len(val_dataset)
        val_losses.append(epoch_val_loss)
        
        print(f"Época {epoch}/{config.EPOCHS} | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")
        
        # Checkpoint: Guardar si es el mejor modelo hasta ahora
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"  -> Guardado nuevo mejor modelo (Val Loss: {best_val_loss:.6f}) en {config.MODEL_SAVE_PATH}")
            
    print("\nEntrenamiento finalizado.")
    print(f"Mejor pérdida de validación lograda: {best_val_loss:.6f}")
    
    # 6. Graficar y guardar la curva de aprendizaje
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, config.EPOCHS + 1), train_losses, label="Pérdida Entrenamiento (MSE)")
    plt.plot(range(1, config.EPOCHS + 1), val_losses, label="Pérdida Validación (MSE)")
    plt.xlabel("Época")
    plt.ylabel("Pérdida (MSE)")
    plt.title("Curva de Aprendizaje - Estimador de Mirada EyeStim")
    plt.legend()
    plt.grid(True)
    
    # Guardar gráfico
    plt.savefig(config.LEARNING_CURVE_PATH)
    plt.close()
    print(f"Curva de aprendizaje exportada a: {config.LEARNING_CURVE_PATH}")

if __name__ == "__main__":
    train_model()
