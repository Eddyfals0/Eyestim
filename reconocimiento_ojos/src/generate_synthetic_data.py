import os
from PIL import Image, ImageDraw
import random

def generar_imagen(tipo, ruta, num_imagenes):
    """
    Genera imágenes sintéticas (dibujadas por código) para que no tengas 
    que descargar nada de internet en este primer experimento.
    """
    for i in range(num_imagenes):
        # Crear un lienzo en blanco (RGB) de 64x64
        imagen = Image.new('RGB', (64, 64), color=(255, 255, 255))
        dibujo = ImageDraw.Draw(imagen)
        
        # Color aleatorio
        color = (random.randint(0, 200), random.randint(0, 200), random.randint(0, 200))

        if tipo == 'ojo':
            # Vamos a simular que un "ojo" es un CÍRCULO
            # Dibujamos un círculo estirado (elipse) al azar
            x0 = random.randint(5, 15)
            y0 = random.randint(20, 30)
            x1 = random.randint(45, 55)
            y1 = random.randint(35, 45)
            dibujo.ellipse([x0, y0, x1, y1], fill=color, outline=(0, 0, 0))
            
            # Dibujamos la pupila
            dibujo.ellipse([27, 27, 37, 37], fill=(0, 0, 0))
            
        else:
            # Vamos a simular que un "no-ojo" es un CUADRADO u otra cosa
            x0 = random.randint(10, 20)
            y0 = random.randint(10, 20)
            x1 = random.randint(40, 50)
            y1 = random.randint(40, 50)
            dibujo.rectangle([x0, y0, x1, y1], fill=color, outline=(0, 0, 0))

        # Guardar en su carpeta correspondiente
        imagen.save(os.path.join(ruta, f"sintetico_{i+1}.jpg"))

if __name__ == "__main__":
    print("🎨 Generando imágenes sintéticas de prueba...")
    
    # Crear carpetas por si acaso
    os.makedirs("../data/train/ojo", exist_ok=True)
    os.makedirs("../data/train/sin_ojo", exist_ok=True)
    os.makedirs("../data/val/ojo", exist_ok=True)
    os.makedirs("../data/val/sin_ojo", exist_ok=True)

    # Entrenar el modelo con 150 fotos de cada uno (Train)
    generar_imagen('ojo', '../data/train/ojo', 150)
    generar_imagen('sin_ojo', '../data/train/sin_ojo', 150)
    
    # Examinarlo (Val) con 30 fotos nuevas de cada uno
    generar_imagen('ojo', '../data/val/ojo', 30)
    generar_imagen('sin_ojo', '../data/val/sin_ojo', 30)

    print("✅ ¡Listo! Carpetas llenas. Hay 300 imágenes para entrenamiento y 60 para validación.")
    print("Ya puedes ejecutar: python src/train.py")
