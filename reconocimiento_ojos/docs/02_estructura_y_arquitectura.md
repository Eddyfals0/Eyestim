# 🏗️ Estructura y Arquitectura del Proyecto

Para que un proyecto de Machine Learning sea limpio, se debe separar muy bien los datos, el código y los resultados (los modelos generados). Aquí te explico cómo hemos estructurado tu proyecto en la carpeta `reconocimiento_ojos`:

```text
reconocimiento_ojos/
│
├── data/                   <-- Aquí irán todas tus imágenes
│   ├── train/              <-- Imágenes para ENTRENAR al modelo
│   │   ├── ojos/           <-- Carpeta con puras fotos de ojos
│   │   └── sin_ojos/       <-- Carpeta con fotos de narices, bocas, paisajes, etc.
│   └── val/                <-- Imágenes de PRUEBA (para ver si aprendió de verdad)
│       ├── ojos/
│       └── sin_ojos/
│
├── docs/                   <-- Las guías que estás leyendo en este momento
│
├── models/                 <-- Aquí se guardará tu IA una vez que esté lista (archivos .pth)
│
├── src/                    <-- ¡El corazón del proyecto! El código fuente en Python.
│   ├── model.py            <-- El plano arquitectónico de tu Red Neuronal (CNN)
│   ├── dataset.py          <-- Código encargado de cargar las imágenes desde /data y pasárselas al modelo
│   ├── train.py            <-- El archivo que ejecutas para que el modelo comience a estudiar las imágenes
│   └── predict.py          <-- El archivo que usarás para cargar una imagen nueva y que la IA te diga si es ojo o no
│
└── requirements.txt        <-- Lista de librerías de Python necesarias para correr el proyecto
```

## Entendiendo el flujo de los scripts (`src/`)

1. **`dataset.py`**: Pide las imágenes de la carpeta `data/`. Les aplica zoom o recortes para que todas sean exactamente de `64x64 pixeles` (a las IA no les gustan las imágenes de diferentes tamaños). Luego las transforma a números matemáticos (Tensores).
2. **`model.py`**: Define la clase de tu IA. Tiene capas de neuronas que actúan como "filtros visuales". Su única función es aceptar los píxeles y devolver un número del 0 al 1 (cercano a 1 es ojo, cercano a 0 es no-ojo).
3. **`train.py`**: Es el maestro director. Ejecuta un bucle: Toma las imágenes de `dataset.py`, las avienta dentro de `model.py`, verifica si se equivocó, lo castiga/premia, e intenta de nuevo. Cuando acaba, escupe un archivo dentro de `models/`.
4. **`predict.py`**: El probador final. Tú descargas una imagen de internet, se la pasas a este script, y el script llama a tu modelo entrenado en `models/` para darte el veredicto.

---
**Siguiente Paso:** Lee el archivo `03_preparacion_datos.md` para saber cómo alistarle la comida a nuestra red neuronal (tus propias imágenes).
