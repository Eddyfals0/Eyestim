# Dictamen de datasets y validez de investigación de EyeStim

**Fecha de auditoría inicial:** 2026-09-05  
**Actualización con entrenamiento:** 2026-09-06  
**Rama evaluada:** `003-pupilometry-attention`  
**Dictamen:** BioID no es adecuado como dataset principal y el sistema actual no valida un “porcentaje de atención”. LPW y EyeDentify ya fueron incorporados; el resultado culminatorio está en [`project_culmination_report.md`](project_culmination_report.md).

## 1. Propósito que realmente puede investigarse

EyeStim mezcla tres problemas distintos:

1. localizar/segmentar la pupila en imágenes de una webcam;
2. estimar la dirección o el punto de mirada en pantalla;
3. inferir atención a una tarea a partir de series temporales de mirada, pupila y conducta.

Los dos primeros son problemas de visión por computadora con una referencia geométrica observable. El tercero es un constructo latente: la pupila también cambia por luminancia, esfuerzo, sorpresa, activación autonómica y fatiga. Hasta que exista validación supervisada contra una condición experimental y conducta observable, la salida debe denominarse **índice heurístico de estabilidad ocular**, no “porcentaje de atención cognitiva”.

## 2. Dictamen sobre BioID

La descripción oficial de BioID indica que se creó para comparar algoritmos de detección facial. Sus 1,521 imágenes incluyen posiciones manuales de los ojos, no elipses de pupila, diámetro, vector de mirada ni etiquetas de atención.

La auditoría local encontró:

| Comprobación | Resultado |
|---|---:|
| Imágenes PGM | 1,521 |
| Anotaciones `.eye` | 1,521 |
| Pares completos | 1,521 |
| Anotaciones fuera de límites | 0 |
| Resolución | 384 × 286 |
| Elipse/diámetro pupilar | No disponible |
| Objetivo de mirada en pantalla | No disponible |
| Condición o desempeño de atención | No disponible |

BioID tiene integridad interna para detección facial/ocular, pero no puede validar el objetivo central de EyeStim. El loader actual agrava el problema: al centrar cada recorte con la propia etiqueta, el objetivo de evaluación siempre es `(0, 0)`.

## 3. Bundle de validación incorporado

No existe un único dataset público que reúna webcam RGB, diámetro de referencia, mirada a pantalla, luminancia y atención experimental. Se incorporaron fuentes complementarias bajo `data/external/` (ignorado por Git):

| Fuente | Uso en esta auditoría | Cobertura descargada | Licencia/condición |
|---|---|---|---|
| Świrski/Cambridge | Elipses reales para diámetro y centro | Ojo izquierdo de 2 participantes; 300 elipses | Citar publicación; la página no declara licencia explícita |
| LPW | Stress test del centro pupilar del CNN | 1 vídeo, 2,000 frames anotados | CC BY-NC-SA 4.0 |
| NEMAR `nm000150` / BBBD | Condición atenta frente a distraída con pupila y mirada | `stim01`, 27 sujetos, sesiones 01/02, 51 grabaciones | CC BY 4.0 |

El descargador reproducible es `scripts/download_validation_data.py`; registra y verifica los checksums publicados cuando están disponibles.

## 4. Resultados de las pruebas ejecutadas

### 4.1 CNN de localización sobre LPW

| Métrica | Resultado |
|---|---:|
| Frames | 2,000 |
| Error medio en ROI 64×64 | 14.415 px |
| Mediana | 14.823 px |
| Percentil 95 | 22.703 px |
| Baseline que siempre predice el centro | 14.415 px |
| Desviación de predicciones X / Y | 0.000002 / 0.000090 |

El modelo guardado es indistinguible de una predicción constante del centro. LPW usa una cámara ocular infrarroja y por eso este resultado es un stress test fuera de dominio, no una cifra final de webcam; aun así, confirma el colapso observado en el entrenamiento.

### 4.2 Pupilometría clásica con elipses reales

El participante 1 se reservó para el barrido de parámetros y el participante 2 como prueba independiente.

| Configuración | Detección | MAE diámetro | Sesgo | Spearman |
|---|---:|---:|---:|---:|
| Configuración inicial (`crop=32`, `offset=15`), participante 1 | 94.7% | 2.361 px | -2.361 px | 0.453 |
| Configuración inicial, participante 2 reservado | 91.3% | 2.744 px | -2.744 px | 0.795 |
| Ajustada (`crop=32`, `offset=35`), participante 2 | 100% | 0.919 px | -0.790 px | 0.808 |

El ajuste mejora mucho el algoritmo, pero no alcanza el valor de referencia de 0.8 px en el sujeto reservado. Al oscurecer 40 niveles el conjunto de prueba, el MAE aumenta a **6.992 px**, lo que confirma una fuerte sensibilidad a iluminación/recorte tonal. La versión experimental adoptó el offset 35 después de integrar el centro LPW y volvió a probarlo de extremo a extremo en el participante 2 (MAE 0.876 px); esto no se declara resuelto ni sustituye una cohorte webcam de repetibilidad controlada.

### 4.3 Validez del score de atención sobre BBBD/NEMAR

Se reprodujo la calibración de 100 muestras y las reglas de `AttentionTracker` en 24 participantes con ambas condiciones. EyeLink entrega área pupilar en unidades arbitrarias; se aplicó `sqrt(área)` como proxy dimensional de diámetro antes de usar los umbrales relativos de EyeStim.

| Métrica | Atenta | Distraída | Diferencia atenta−distraída | Wilcoxon p |
|---|---:|---:|---:|---:|
| Score EyeStim medio | 80.734 | 86.616 | **-5.882** | **0.000430** |
| Cambio del diámetro proxy desde baseline | -6.990% | -2.786% | -4.205 pp | 0.000494 |
| Dispersión de mirada | 6.381° | 6.754° | -0.373° | 0.359626 |

Sólo 5 de 24 participantes (20.8%) presentaron el score en la dirección esperada. El score fue significativamente **mayor en la condición distraída**. Un clasificador exploratorio con cuatro métricas y validación `leave-one-subject-out` obtuvo AUC 0.696 y balanced accuracy 0.641: las señales contienen cierta información, pero la fórmula fija actual la combina en la dirección equivocada y no está calibrada como probabilidad.

## 5. Dataset recomendado e integración realizada

Para entrenar el componente de diámetro en el mismo dominio de la aplicación, la opción primaria es **EyeDentify/EyeDentify++**: imágenes webcam sincronizadas con diámetro de referencia Tobii, 51 participantes y particiones de validación por participante. Su descarga se distribuye mediante Kaggle y requiere aceptar sus condiciones CC BY-NC 4.0.

Esta recomendación fue implementada: se extrajeron 9,425 imágenes que cubren los 51 participantes, se entrenó un regresor de diámetro y se reservó la prueba por participante. LPW también sustituyó a BioID como fuente activa para el centro pupilar. Las métricas, checkpoints y curvas están documentados en el informe culminatorio.

Para la mirada en laptop debe utilizarse **MPIIFaceGaze/MPIIGaze**, que incluye posiciones reales de mirada, variación de iluminación y participantes identificables para evaluación `leave-one-person-out`. Su uso está restringido a investigación científica no comercial.

Para validar atención específica a contenido educativo, **BBBD/NEMAR `nm000150`** es mejor que BioID porque tiene una manipulación atenta/distraída, mirada, pupila y resultados conductuales. No contiene frames webcam y por ello valida el constructo, no el detector visual.

La solución científicamente correcta es un pipeline de datasets, no sustituir BioID por un solo archivo:

1. EyeDentify para diámetro pupilar desde webcam;
2. MPIIFaceGaze para mirada a pantalla;
3. BBBD/NEMAR y una recolección local para atención y generalización al dispositivo final.

## 6. Protocolo mínimo para una afirmación válida

- Separar entrenamiento, validación y prueba por **participante**, nunca por frames.
- Registrar diámetro de referencia en mm, gaze X/Y, head pose, parpadeo/confianza, luminancia de pantalla y ambiental, distancia a cámara y timestamp sincronizado.
- Contrabalancear condición atenta/distraída y orden de estímulos.
- Añadir desempeño observable: respuestas correctas, tiempo de reacción y *probes* de mind-wandering.
- Evaluar pupila con MAE en mm, Bland–Altman, correlación y tasa de detección.
- Evaluar mirada con error angular, error de punto y F1 por zona.
- Evaluar atención con AUROC, balanced accuracy, sensibilidad/especificidad y calibración; no llamarla porcentaje hasta verificar una curva de calibración en sujetos no vistos.
- Reportar intervalos de confianza por bootstrap de participantes y resultados por iluminación, lentes, tono de piel/iris y dispositivo.
- Ejecutar un piloto para estimar varianza y calcular potencia antes de fijar el tamaño final; una meta inicial razonable es al menos 30 participantes completos para el piloto, no como garantía de potencia.

## 7. Fuentes primarias

- BioID: https://www.bioid.com/face-database/
- Świrski/Cambridge: https://www.cl.cam.ac.uk/research/rainbow/projects/pupiltracking/datasets/
- LPW: https://doi.org/10.18419/DARUS-3237
- MPIIGaze: https://collaborative-ai.org/research/datasets/MPIIGaze/
- EyeDentify: https://github.com/vijulshah/eyedentify
- BBBD/NEMAR: https://www.nemar.org/dataset/nm000150
- Revisión de dilatación pupilar y esfuerzo: https://doi.org/10.3758/s13423-018-1432-y
- Revisión sobre inferencia cognitiva desde la pupila: https://pmc.ncbi.nlm.nih.gov/articles/PMC7271902/
