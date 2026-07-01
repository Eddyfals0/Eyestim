# EyeStim: Analizador de Atención Visual y Estímulos Adaptativos
## Documento de Alcance y Necesidad del Proyecto

### 1. Contexto y Necesidad del Proyecto
En la actualidad, las plataformas digitales de entretenimiento e interacción (como TikTok, Instagram y YouTube) emplean complejos algoritmos de Inteligencia Artificial para captar, analizar y retener la atención de los usuarios. Sin embargo, estas tecnologías suelen operar como "cajas negras" y sus fines son predominantemente comerciales, sin un escrutinio académico detallado sobre cómo se ve afectada la concentración, la curiosidad o el interés de un sujeto.

La propuesta **EyeStim** surge como una iniciativa académica de investigación en el **Centro de Modelación y Simulación de Sistemas** de la Facultad de Ingeniería. Su necesidad radica en:
*   **Estudio Ético y Controlado**: Analizar científicamente y de forma no invasiva cómo reacciona el sistema visual humano ante diferentes estímulos.
*   **Decodificación del Interés**: Identificar qué elementos (colores, ritmos, movimientos) capturan más eficazmente la atención humana, sentando las bases para crear estímulos educativos y de comunicación visual más eficientes.
*   **Aporte Metodológico**: Desarrollar un modelo que sirva a futuras investigaciones de sistemas complejos de interacción persona-computador.

---

### 2. Alcance del MVP (Módulo de Visión)
Para mantener el foco en la investigación científica y evitar la complejidad de desarrollo de una aplicación web/móvil completa, el proyecto se centrará en construir un **MVP del modelo de visión**. Este modelo operará en tiempo real y tendrá el siguiente alcance funcional:

1.  **Detección de Rostro y Ojos**:
    *   Abrir la cámara web del dispositivo del usuario.
    *   Detectar de forma robusta la ubicación del rostro y extraer la región de interés (ROI) correspondiente a ambos ojos en tiempo real.
2.  **Seguimiento del Iris e Iris/Pupila (Retina)**:
    *   Localizar el centro exacto de la pupila/iris en cada ojo.
    *   Estimar dinámicamente el tamaño/diámetro de la retina/pupila (útil para medir respuestas de dilatación asociadas al interés o la sorpresa).
3.  **Estimación de la Dirección de la Mirada**:
    *   Calcular y trazar una línea vectorial indicadora que señale la dirección estimada hacia donde el usuario está apuntando la mirada respecto al centro de su ojo.
4.  **Simplicidad y Enfoque Experimental**:
    *   Todo el desarrollo se estructurará con fines de investigación, permitiendo realizar pruebas experimentales y documentar los resultados en formatos sencillos y legibles.

---

### 3. Objetivos del MVP de Visión
*   **Registrar el comportamiento visual**: Guardar o visualizar las coordenadas de interés y los cambios en el tamaño de la pupila de forma interactiva.
*   **Evitar redundancia**: Limpiar el repositorio de código duplicado, apps adicionales o scripts de prueba obsoletos, manteniendo un único núcleo de visión bien estructurado y fácil de ejecutar.
