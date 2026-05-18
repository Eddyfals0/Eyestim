from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def preparar_datos(ruta_datos="../data", tamano_lote=32):
    """
    Lee las imágenes de las carpetas, las arregla para que todas sean del mismo tamaño,
    y las envuelve en un 'DataLoader' que las entregará por puñados (lotes/batches) a la IA.
    """
    
    # 1. Definir cómo queremos transformar las fotos
    transformaciones = transforms.Compose([
        transforms.Resize((64, 64)),        # TODAS las fotos se estirarán a 64x64 pixeles
        transforms.Grayscale(num_output_channels=1), # Convertir a escala de grises
        transforms.ToTensor(),              # Convierte la imagen a números flotantes que PyTorch entiende
        # Normalizamos los colores para que sea más matemáticamente estable:
        transforms.Normalize(mean=[0.5], std=[0.5]) 
    ])

    # 2. Leer las carpetas mágicamente con ImageFolder
    # Automáticamente etiqueta la carpeta 'ojo' como clase 0 y 'sin_ojo' como clase 1 (orden alfabético)
    try:
        dataset_entrenamiento = ImageFolder(root=f"{ruta_datos}/train", transform=transformaciones)
        dataset_validacion = ImageFolder(root=f"{ruta_datos}/val", transform=transformaciones)
    except FileNotFoundError as e:
        print("¡ERROR! No tienes tus imágenes en la carpeta 'data/'.")
        print("Por favor crea las carpetas: data/train/ojo, data/train/sin_ojo, etc. y pon fotos dentro.")
        raise e

    clases_detectadas = dataset_entrenamiento.classes # ejemplo: ['ojo', 'sin_ojo']
    
    # Si hay muchas carpetas (por ejemplo: 'forward_look', 'left_look', 'ojo', 'sin_ojo'),
    # queremos que TODAS sean consideradas "ojo" (clase 0), EXCEPTO "sin_ojo" (clase 1).
    if 'sin_ojo' in clases_detectadas:
        idx_sin_ojo = clases_detectadas.index('sin_ojo')
    else:
        idx_sin_ojo = -1 # Fallback, asumimos ojo todo el rato
        
    def transformador_etiquetas(indice_clase_original):
        return 1.0 if indice_clase_original == idx_sin_ojo else 0.0
        
    dataset_entrenamiento.target_transform = transformador_etiquetas
    dataset_validacion.target_transform = transformador_etiquetas

    # 3. Crear los cargadores de datos (DataLoaders)
    # Entregará imágenes en grupos de (tamano_lote), p.e. de 32 en 32 fotos al mismo tiempo.
    # shuffle=True hace que el modelo vea ejemplos mezclados, lo que mejora el aprendizaje.
    cargador_entrenamiento = DataLoader(dataset_entrenamiento, batch_size=tamano_lote, shuffle=True)
    cargador_validacion = DataLoader(dataset_validacion, batch_size=tamano_lote, shuffle=False)

    return cargador_entrenamiento, cargador_validacion, clases_detectadas
