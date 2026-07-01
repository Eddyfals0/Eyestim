import torch
import torch.nn as nn

class EyePupilCNN(nn.Module):
    """
    Arquitectura convolucional personalizada de bajo nivel en PyTorch
    para la regresión bidimensional del centro de la pupila.
    """
    def __init__(self):
        super(EyePupilCNN, self).__init__()
        
        # Bloque Convolucional 1 (Entrada: 1x64x64 -> Salida: 16x32x32)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bloque Convolucional 2 (Entrada: 16x32x32 -> Salida: 32x16x16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bloque Convolucional 3 (Entrada: 32x16x16 -> Salida: 64x8x8)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Capas densas de regresión
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.relu_fc = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        # Capa de salida con activación Tanh para restringir a [-1.0, 1.0]
        self.fc2 = nn.Linear(128, 2)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        
        x = self.flatten(x)
        x = self.dropout(self.relu_fc(self.fc1(x)))
        x = self.tanh(self.fc2(x))
        return x

if __name__ == "__main__":
    # Test unitario básico del modelo
    model = EyePupilCNN()
    dummy_input = torch.zeros((4, 1, 64, 64))  # Lote de 4 imágenes
    output = model(dummy_input)
    print("=== Test Unitario del Modelo ===")
    print(f"Dimensiones de entrada: {dummy_input.shape}")
    print(f"Dimensiones de salida: {output.shape}")
    print(f"Rango de valores de salida (min/max): {output.min().item():.4f} / {output.max().item():.4f}")
    assert output.shape == (4, 2), "Error en las dimensiones de salida."
    print("Modelo definido e inicializado correctamente.")
