# 🧠 Introducción a Modelos de IA para Imágenes

¡Hola! Si apenas estás comenzando en este mundo, ¡no te preocupes! Entrenar un modelo de Inteligencia Artificial para reconocer si hay un "ojo" o no en una imagen es un excelente primer paso. 

Para este proyecto usaremos **PyTorch**, el estándar actual en la industria y la investigación para IA.

## 1. ¿Cómo \"aprende\" una máquina a ver?

Las computadoras no ven imágenes como nosotros; ven matrices (tablas) enormes de números puros que representan los colores de los píxeles (Rojo, Verde, Azul). Para ayudar a la computadora a encontrar patrones visuales (como la forma circular del iris o la línea de las pestañas), usamos un tipo de modelo llamado **Red Neuronal Convolucional (CNN)**.

## 2. Red Neuronal Convolucional (CNN)

Imagina que tienes una lupa y la vas pasando por toda la imagen en pequeños cuadros. 
- Las primeras veces que pasas la lupa, el modelo prende a reconocer cosas súper básicas: líneas, bordes, o simples contrastes.
- A medida que pasas la información a capas más profundas de la red, esas líneas se combinan para formar figuras geométricas (círculos, curvas).
- Al final, el modelo es capaz de reconocer características complejas: ¡Un ojo!

## 3. ¿Cómo entrenamos el modelo? (El Aprendizaje Supervisado)

Para que el modelo aprenda a reconocer ojos, necesitamos usar algo llamado **Aprendizaje Supervisado**. Consiste en actuar como un maestro con miles de tarjetas de estudio.

1. **Le muestras una imagen** que tiene un ojo.
2. **El modelo adivina** (al principio al azar): *"Creo que NO hay un ojo aquí"*.
3. **Tú lo corriges** dándole la respuesta correcta (esto se llama **Label** o Etiqueta): *"¡Te equivocaste! Esta imagen SÍ es un ojo"*.
4. **Cálculo de Error (Loss)**: El modelo calcula qué tan equivocado estaba usando una "Función de Pérdida".
5. **Ajuste (Backpropagation y Optimizador)**: El modelo ajusta sus engranes internos (pesos matemáticos) para no volver a equivocarse la próxima vez que vea esa misma imagen.

Este ciclo iterativo se repite miles de veces a través de tus imágenes. Cada vez que le damos una vuelta a todas tus imágenes, lo llamamos una **Época (Epoch)**.

---
**Siguiente Paso:** Lee el archivo `02_estructura_y_arquitectura.md` para entender cómo está organizado el código que va a hacer todo esto posible.
