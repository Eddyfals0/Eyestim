import torch
import torch.nn as nn

class DetectorDeOjosCNN(nn.Module):
    """
    Esta es la 'Arquitectura' de nuestro modelo.
    Aquí definimos cuántos filtros visuales aplicaremos a las imágenes.
    """
    def __init__(self):
        super(DetectorDeOjosCNN, self).__init__()
        
        # 1. CAPAS CONVOLUCIONALES (Los ojos de nuestra IA)
        # La entrada será una imagen en escala de grises (1 canal)
        self.capas_visuales = nn.Sequential(
            # Primer filtro
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(), # Función de activación (enciende la neurona si detecta algo útil)
            nn.MaxPool2d(kernel_size=2, stride=2), # Reduce la imagen a la mitad para procesar más rápido
            
            # Segundo filtro
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Tercer filtro (busca patrones muy complejos)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 2. CAPAS LINEALES (El Cerebro)
        # Después de pasar los filtros, aplanamos los datos y decidimos: ¿Ojo o No Ojo?
        self.cerebro_decisor = nn.Sequential(
            nn.Flatten(), # Aplana los mapas de características a una lista simple de números
            # Calcularemos el tamaño: Asumiendo que la imagen entra de 64x64 pixeles.
            # Tras 3 MaxPools de reducción x2, queda en 8x8 pixeles. (64/2 -> 32/2 -> 16/2 -> 8)
            nn.Linear(in_features=64 * 8 * 8, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.5), # Apagamos neuronas al azar para que no memorice los datos (evita sobreajuste)
            nn.Linear(in_features=128, out_features=1) # 1 neurona al final: Cercano a 1 es un ojo, cercano a 0 es otra cosa
        )

    def forward(self, imagen):
        """
        Esta función describe el viaje de la imagen paso a paso dentro de la red.
        """
        # Pasa por los filtros
        caracteristicas_visuales = self.capas_visuales(imagen)
        # Pasa por el cerebro que toma la decisión
        decision = self.cerebro_decisor(caracteristicas_visuales)
        return decision

# Si ejecutas este archivo solo, simplemente imprimirá cómo se ve tu red por dentro.
if __name__ == "__main__":
    modelo = DetectorDeOjosCNN()
    print("Arquitectura de la Red Neuronal Creada existosamente:")
    print(modelo)
