import os
import torch
import torch.nn as nn
import torch.optim as optim
from model import DetectorDeOjosCNN
from dataset import preparar_datos

# Configuración del entrenamiento MUY rápido
EPOCAS = 1                  
TASA_APRENDIZAJE = 0.001     
RUTA_DATOS = "../data"       

def entrenar_rapido():
    os.makedirs("../models", exist_ok=True)
    cargador_entrenamiento, _, _ = preparar_datos(RUTA_DATOS, tamano_lote=64)

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = DetectorDeOjosCNN().to(dispositivo)
    criterio_error = nn.BCEWithLogitsLoss() 
    optimizador = optim.Adam(modelo.parameters(), lr=TASA_APRENDIZAJE)

    print("Entrenando en modo turbo (1 vuelta parcial)...")
    modelo.train()
    
    # Entrenar solo con el 10% de los datos para dárselo de inmediato
    max_batches = len(cargador_entrenamiento) // 10
    if max_batches == 0: max_batches = 1
    
    for i, (imagenes, etiquetas_reales) in enumerate(cargador_entrenamiento):
        if i >= max_batches:
            break
            
        imagenes = imagenes.to(dispositivo)
        etiquetas_reales = etiquetas_reales.view(-1, 1).float().to(dispositivo)

        optimizador.zero_grad() 
        prediccion = modelo(imagenes) 
        error = criterio_error(prediccion, etiquetas_reales) 
        error.backward() 
        optimizador.step() 

    ruta_guardado = "../models/mi_modelo_detectar_ojos.pth"
    torch.save(modelo.state_dict(), ruta_guardado)
    print("Modelo rápido creado exitosamente.")

if __name__ == "__main__":
    entrenar_rapido()
