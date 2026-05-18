# 📂 Preparación de los Datos (El Dataset)

Como aprendimos en el primer archivo, una Inteligencia Artificial para visión por computadora se alimenta directamente de las imágenes que le entregas. 

En Python (particularmente con `PyTorch`), la manera más universal y sencilla de decirle a la IA "¿Cuál imagen es qué cosa?" no es con un archivo de Excel pesado, sino usando **¡Simples carpetas!**

## La regla de Oro de PyTorch (`ImageFolder`)

PyTorch tiene una herramienta mágica llamada `ImageFolder`. Esta herramienta asume que **el nombre de la carpeta en la que está la foto, es la etiqueta correcta de la foto**. 

Es por esto que nuestro proyecto espera esta estructura exacta:

```text
data/
 ├── train/
 │   ├── ojo/         <--- (ej. ojo_001.jpg, ojo_002.png)
 │   └── sin_ojo/     <--- (ej. perro.jpg, oreja.png, carro.jpg)
 │
 └── val/
     ├── ojo/         <--- (ej. ojo_val_01.jpg)
     └── sin_ojo/     <--- (ej. vaso.jpg)
```

## ¿Qué es "Train" y "Val" (Entrenamiento y Validación)?

1. **La carpeta `train` (Entrenamiento):** Aquí deberás poner alrededor del 80% de tus fotos. El modelo usará ESTAS fotos para aprender. Literarmente "estudiará" estos datos una y otra vez.
   
2. **La carpeta `val` (Validación):** Si a un estudiante le pones el mismo examen con el que estudió, obvio sacará 100 y pensarás que es un genio, pero igual y todo fue pura memoria. Para saber si el modelo de verdad aprendió "el concepto de lo que es un ojo", debes examinarlo con imágenes **QUE JAMÁS HAYA VISTO EN SU VIDA**. A esta prueba la llamamos conjunto de validación (`val`). Pon el 20% restante de imágenes aquí.

## Tus Tareas antes de "entrenar":
1. Ve a la carpeta `data/` en este proyecto (crea las carpetas necesarias como indico arriba).
2. Descarga de internet recortes de ojos y ponlos en las carpetas de `ojo/`.
3. Descarga fotos de cualquier otra cosa (incluso otras partes de la cara como narices, labios) y ponlos en `sin_ojo/`.
4. ¡Asegúrate de tener al menos unas 50-100 imágenes por categoría para empezar a jugar! (Mientras más, siempre es mejor).

Una vez que tengas tus imágenes organizadas así, ¡estás listo para ir a ejecutar tu código!
