"""Build the culminating Spanish academic paper for EyeStim as a PDF."""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib.utils import ImageReader


ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "docs" / "training"
RESEARCH = ROOT / "docs" / "research"
FIGURES = ROOT / "tmp" / "pdfs" / "eyestim_academic_paper"
OUTPUT = ROOT / "output" / "pdf" / "eyestim_paper_academico_2026.pdf"

NAVY = colors.HexColor("#172554")
BLUE = colors.HexColor("#1d4ed8")
TEAL = colors.HexColor("#0f766e")
GREEN = colors.HexColor("#15803d")
ORANGE = colors.HexColor("#c2410c")
RED = colors.HexColor("#b91c1c")
GRAY = colors.HexColor("#475569")
LIGHT_BLUE = colors.HexColor("#eff6ff")
LIGHT_GRAY = colors.HexColor("#f8fafc")
RULE = colors.HexColor("#cbd5e1")


def register_fonts() -> None:
    font_dir = ROOT / ".venv" / "Lib" / "site-packages" / "matplotlib" / "mpl-data" / "fonts" / "ttf"
    fonts = {
        "PaperSerif": "DejaVuSerif.ttf",
        "PaperSerif-Bold": "DejaVuSerif-Bold.ttf",
        "PaperSerif-Italic": "DejaVuSerif-Italic.ttf",
        "PaperSans": "DejaVuSans.ttf",
        "PaperSans-Bold": "DejaVuSans-Bold.ttf",
    }
    for alias, filename in fonts.items():
        path = font_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"No se encontró la fuente requerida: {path}")
        pdfmetrics.registerFont(TTFont(alias, str(path)))
    pdfmetrics.registerFontFamily(
        "PaperSerif",
        normal="PaperSerif",
        bold="PaperSerif-Bold",
        italic="PaperSerif-Italic",
        boldItalic="PaperSerif-Bold",
    )


def make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "PaperTitle",
            parent=base["Title"],
            fontName="PaperSerif-Bold",
            fontSize=20,
            leading=24,
            textColor=NAVY,
            alignment=TA_CENTER,
            spaceAfter=8,
        ),
        "institution": ParagraphStyle(
            "Institution",
            parent=base["Title"],
            fontName="PaperSans-Bold",
            fontSize=18,
            leading=22,
            textColor=NAVY,
            alignment=TA_CENTER,
            spaceAfter=7,
        ),
        "cover_title": ParagraphStyle(
            "CoverTitle",
            parent=base["Title"],
            fontName="PaperSerif-Bold",
            fontSize=21,
            leading=26,
            textColor=NAVY,
            alignment=TA_CENTER,
            spaceAfter=11,
        ),
        "subtitle": ParagraphStyle(
            "PaperSubtitle",
            parent=base["Normal"],
            fontName="PaperSans",
            fontSize=10,
            leading=14,
            textColor=GRAY,
            alignment=TA_CENTER,
            spaceAfter=5,
        ),
        "author": ParagraphStyle(
            "PaperAuthor",
            parent=base["Normal"],
            fontName="PaperSans-Bold",
            fontSize=10,
            leading=13,
            textColor=NAVY,
            alignment=TA_CENTER,
            spaceAfter=8,
        ),
        "abstract_title": ParagraphStyle(
            "AbstractTitle",
            parent=base["Heading2"],
            fontName="PaperSans-Bold",
            fontSize=10,
            leading=12,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=3,
        ),
        "abstract": ParagraphStyle(
            "Abstract",
            parent=base["BodyText"],
            fontName="PaperSerif",
            fontSize=8.6,
            leading=11.2,
            textColor=colors.HexColor("#1e293b"),
            alignment=TA_JUSTIFY,
            spaceAfter=7,
        ),
        "h1": ParagraphStyle(
            "Section",
            parent=base["Heading1"],
            fontName="PaperSans-Bold",
            fontSize=13.5,
            leading=17,
            textColor=NAVY,
            spaceBefore=11,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "Subsection",
            parent=base["Heading2"],
            fontName="PaperSans-Bold",
            fontSize=10.5,
            leading=13,
            textColor=BLUE,
            spaceBefore=8,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "PaperBody",
            parent=base["BodyText"],
            fontName="PaperSerif",
            fontSize=9,
            leading=12.2,
            textColor=colors.HexColor("#111827"),
            alignment=TA_JUSTIFY,
            spaceAfter=5.5,
        ),
        "bullet": ParagraphStyle(
            "PaperBullet",
            parent=base["BodyText"],
            fontName="PaperSerif",
            fontSize=8.8,
            leading=11.5,
            textColor=colors.HexColor("#111827"),
            alignment=TA_JUSTIFY,
            leftIndent=4,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=base["BodyText"],
            fontName="PaperSans",
            fontSize=7.7,
            leading=10,
            textColor=GRAY,
            alignment=TA_CENTER,
            spaceBefore=4,
            spaceAfter=8,
        ),
        "table": ParagraphStyle(
            "TableCell",
            parent=base["BodyText"],
            fontName="PaperSans",
            fontSize=7.2,
            leading=9.2,
            textColor=colors.HexColor("#111827"),
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            parent=base["BodyText"],
            fontName="PaperSans-Bold",
            fontSize=7.2,
            leading=9.2,
            textColor=colors.white,
            alignment=TA_CENTER,
        ),
        "note": ParagraphStyle(
            "PaperNote",
            parent=base["BodyText"],
            fontName="PaperSans",
            fontSize=8.2,
            leading=11,
            textColor=NAVY,
            backColor=LIGHT_BLUE,
            borderColor=colors.HexColor("#93c5fd"),
            borderWidth=0.8,
            borderPadding=7,
            spaceBefore=5,
            spaceAfter=8,
        ),
        "warning": ParagraphStyle(
            "PaperWarning",
            parent=base["BodyText"],
            fontName="PaperSans",
            fontSize=8.2,
            leading=11,
            textColor=colors.HexColor("#7f1d1d"),
            backColor=colors.HexColor("#fef2f2"),
            borderColor=colors.HexColor("#fca5a5"),
            borderWidth=0.8,
            borderPadding=7,
            spaceBefore=5,
            spaceAfter=8,
        ),
        "reference": ParagraphStyle(
            "Reference",
            parent=base["BodyText"],
            fontName="PaperSerif",
            fontSize=7.6,
            leading=10,
            textColor=colors.HexColor("#1f2937"),
            leftIndent=14,
            firstLineIndent=-14,
            spaceAfter=3.5,
        ),
    }


def paragraph(text: str, styles: dict, style: str = "body") -> Paragraph:
    return Paragraph(text, styles[style])


def bullets(items: list[str], styles: dict) -> ListFlowable:
    return ListFlowable(
        [ListItem(Paragraph(item, styles["bullet"]), leftIndent=12) for item in items],
        bulletType="bullet",
        bulletFontName="PaperSans",
        bulletFontSize=6,
        leftIndent=18,
        bulletColor=BLUE,
        spaceAfter=5,
    )


def academic_table(data: list[list[str]], widths: list[float], styles: dict) -> Table:
    converted = []
    for row_index, row in enumerate(data):
        style = styles["table_header"] if row_index == 0 else styles["table"]
        converted.append([Paragraph(str(cell), style) for cell in row])
    table = Table(converted, colWidths=widths, repeatRows=1, hAlign="CENTER")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
                ("GRID", (0, 0), (-1, -1), 0.35, RULE),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def scaled_image(path: Path, width: float, max_height: float) -> Image:
    image_width, image_height = ImageReader(str(path)).getSize()
    scale = min(width / image_width, max_height / image_height)
    return Image(str(path), width=image_width * scale, height=image_height * scale)


def figure(path: Path, caption: str, styles: dict, width: float = 16.2 * cm, max_height: float = 8.7 * cm) -> KeepTogether:
    return KeepTogether(
        [
            scaled_image(path, width, max_height),
            Paragraph(caption, styles["caption"]),
        ]
    )


def page_decor(canvas, doc) -> None:
    canvas.saveState()
    page = canvas.getPageNumber()
    canvas.setTitle("EyeStim: resultados de Servicio Social y pupilometría híbrida")
    canvas.setAuthor("Eduardo Guerra Bedolla")
    canvas.setSubject("Datos, modelos, entrenamiento, resultados y validación")
    canvas.setKeywords("EyeStim, pupilometría, LPW, EyeDentify, CNN, OpenCV")
    if page > 1:
        canvas.setFont("PaperSans", 7.2)
        canvas.setFillColor(GRAY)
        canvas.drawString(1.8 * cm, A4[1] - 1.15 * cm, "BUAP - RESULTADOS DE SERVICIO SOCIAL")
        canvas.drawRightString(A4[0] - 1.8 * cm, A4[1] - 1.15 * cm, "PROYECTO EYESTIM")
        canvas.setStrokeColor(RULE)
        canvas.line(1.8 * cm, A4[1] - 1.32 * cm, A4[0] - 1.8 * cm, A4[1] - 1.32 * cm)
    canvas.setStrokeColor(RULE)
    canvas.line(1.8 * cm, 1.22 * cm, A4[0] - 1.8 * cm, 1.22 * cm)
    canvas.setFont("PaperSans", 7.2)
    canvas.setFillColor(GRAY)
    canvas.drawString(1.8 * cm, 0.86 * cm, "Repositorio: github.com/Eddyfals0/Eyestim")
    canvas.drawRightString(A4[0] - 1.8 * cm, 0.86 * cm, f"Página {page}")
    canvas.restoreState()


def build() -> Path:
    register_fonts()
    styles = make_styles()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    diameter = json.loads((TRAINING / "training_summary.json").read_text(encoding="utf-8"))
    center = json.loads((TRAINING / "center_training_summary.json").read_text(encoding="utf-8"))
    final = json.loads((TRAINING / "final_validation_summary.json").read_text(encoding="utf-8"))
    end_to_end = final["cambridge_end_to_end_held_out"]
    reporter = final["reporter_benchmark"]

    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=1.8 * cm,
        leftMargin=1.8 * cm,
        topMargin=1.65 * cm,
        bottomMargin=1.55 * cm,
        title="EyeStim: resultados de Servicio Social y validación de pupilometría híbrida",
        author="Eduardo Guerra Bedolla",
    )
    usable_width = A4[0] - 3.6 * cm
    story = []

    story.append(Spacer(1, 0.55 * cm))
    story.append(paragraph("BENEMÉRITA UNIVERSIDAD AUTÓNOMA DE PUEBLA", styles, "institution"))
    story.append(paragraph("Facultad de Ciencias de la Computación", styles, "author"))
    story.append(paragraph("Av. San Claudio s/n, C.U., Col. San Manuel, Puebla, México", styles, "subtitle"))
    story.append(Spacer(1, 1.0 * cm))
    story.append(HRFlowable(width="100%", thickness=2.2, color=BLUE, spaceBefore=4, spaceAfter=16))
    story.append(paragraph("DOCUMENTO DE RESULTADOS DE SERVICIO SOCIAL", styles, "author"))
    story.append(paragraph("EyeStim: Analizador de atención visual y generador de estímulos audiovisuales adaptativos", styles, "cover_title"))
    story.append(paragraph(
        "Programa: Algoritmos de Inteligencia Artificial Aplicado al Análisis de Sistemas",
        styles,
        "note",
    ))
    cover_data = [
        ["Estudiante", "Eduardo Guerra Bedolla"],
        ["Matrícula", "202231161"],
        ["Centro receptor", "Centro de Modelación y Simulación de Sistemas, Facultad de Ingeniería"],
        ["Responsable del programa", "M.E.S. Gabriela Yáñez Pérez - Docente"],
        ["Correo", "gabriela.yanez@correo.buap.mx"],
        ["Ubicación", "Facultad de Ingeniería, edificio ING2, cubículo 207"],
    ]
    cover_table = Table(
        [[Paragraph(f"<b>{left}</b>", styles["table"]), Paragraph(right, styles["table"])] for left, right in cover_data],
        colWidths=[4.5 * cm, 11.5 * cm],
        hAlign="CENTER",
    )
    cover_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), LIGHT_BLUE),
        ("ROWBACKGROUNDS", (1, 0), (1, -1), [colors.white, LIGHT_GRAY]),
        ("GRID", (0, 0), (-1, -1), 0.45, RULE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(cover_table)
    story.append(Spacer(1, 1.25 * cm))
    story.append(paragraph("Propuesta original entregada el 20 de octubre de 2025", styles, "author"))
    story.append(paragraph("Heroica Puebla de Zaragoza", styles, "subtitle"))
    story.append(Spacer(1, 0.35 * cm))
    story.append(paragraph(
        "Aclaración sobre las fechas: el 20 de octubre de 2025 corresponde a la propuesta de Servicio Social. El entrenamiento y las pruebas técnicas se realizaron el 6 de septiembre de 2026, y este informe se terminó el 9 de septiembre de 2026.",
        styles,
        "warning",
    ))
    story.append(PageBreak())

    story.append(Spacer(1, 0.25 * cm))
    story.append(paragraph("INFORME ACADÉMICO DE RESULTADOS", styles, "subtitle"))
    story.append(paragraph("EyeStim: desarrollo y evaluación de un sistema de pupilometría con webcam", styles, "title"))
    story.append(paragraph("Eduardo Guerra Bedolla · Matrícula 202231161 · Benemérita Universidad Autónoma de Puebla", styles, "author"))
    story.append(paragraph("9 de septiembre de 2026 · Procesamiento local en CPU", styles, "subtitle"))
    story.append(HRFlowable(width="100%", thickness=1.2, color=BLUE, spaceBefore=5, spaceAfter=10))

    story.append(paragraph("Resumen", styles, "abstract_title"))
    story.append(paragraph(
        "EyeStim fue pensado como un sistema de pupilometría y análisis ocular en tiempo real mediante webcam. La primera versión entrenaba un modelo de centro con BioID, un conjunto para detección facial que no contiene posiciones precisas de pupila, diámetro ni atención. Por este problema, el modelo casi siempre predecía el mismo punto. La investigación se reorganizó para estudiar por separado el centro, el diámetro y la posible atención. El centro se entrenó de nuevo con 16,000 imágenes de Labelled Pupils in the Wild (LPW); el diámetro se entrenó con 9,425 recortes EyeDentify de 51 participantes y medidas Tobii; Cambridge/Świrski se usó para probar el proceso completo CNN-elipse; y BBBD/NEMAR permitió revisar si el índice ocular podía interpretarse como atención. Los participantes de entrenamiento, validación y prueba fueron distintos. El modelo LPW redujo el error medio de centro de 10.374 a 4.445 px. El modelo EyeDentify obtuvo MAE 0.202 mm frente a 0.224 mm al predecir siempre la media, aunque la comparación de los siete participantes de prueba no fue suficiente para asegurar una mejora general (p=0.296875). La prueba Cambridge logró 100% de detección, MAE de diámetro de 0.876 px y Spearman de 0.785. El índice 0-100 fue mayor en la condición distraída, por lo que no puede presentarse como una medida de atención. Se concluye que EyeStim es útil como prototipo experimental de pupilometría, pero todavía no es un instrumento clínico ni un medidor validado de mirada o atención.",
        styles,
        "abstract",
    ))
    story.append(paragraph("Palabras clave: pupilometría; estimación de centro pupilar; diámetro pupilar; webcam; LPW; EyeDentify; validación por participante; visión por computadora.", styles, "abstract"))
    story.append(figure(FIGURES / "figure_1_pipeline.png", "Figura 1. Funcionamiento general. LPW y EyeDentify resuelven objetivos diferentes; OpenCV mejora el ajuste de la pupila y el índice temporal se conserva sólo como una salida experimental.", styles, width=usable_width, max_height=6.1 * cm))
    story.append(PageBreak())

    story.append(paragraph("1. Contexto del Servicio Social y alcance de la propuesta", styles, "h1"))
    story.append(paragraph(
        "EyeStim se desarrolló dentro del programa <i>Algoritmos de Inteligencia Artificial Aplicado al Análisis de Sistemas</i> del Centro de Modelación y Simulación de Sistemas. La propuesta surgió al observar que plataformas como Instagram, TikTok y YouTube usan algoritmos para conocer preferencias y mantener la atención. El proyecto planteó estudiar el comportamiento visual de forma controlada, local, no invasiva y responsable, con posibles aplicaciones en educación, comunicación visual y simulación.",
        styles,
    ))
    story.append(paragraph(
        "El objetivo general fue analizar la atención visual ante estímulos audiovisuales, identificar factores relacionados con el interés o la curiosidad y preparar las bases para diseñar estímulos más efectivos de manera ética. El trabajo permitió construir y probar la captura y medición de la pupila. Sin embargo, todavía no hay evidencia suficiente para afirmar atención, emoción o curiosidad, y no se implementó un generador automático de estímulos. Esta diferencia entre lo propuesto y lo alcanzado se indica de manera clara.",
        styles,
    ))
    objective_table = [
        ["Objetivo específico de la propuesta", "Estado", "Resultado verificable"],
        ["Observar y registrar comportamiento visual ante imágenes o vídeos", "Cumplido como prototipo", "Captura por webcam, recorte del ojo, series de centro/diámetro y reporte de sesión"],
        ["Identificar patrones de atención e impacto visual", "Parcial", "Se obtuvieron patrones oculares; las pruebas mostraron que el índice actual no mide atención"],
        ["Analizar factores emocionales o perceptivos", "No validado", "No existen etiquetas emocionales ni protocolo experimental propio"],
        ["Diseñar estímulos audiovisuales que incrementen atención", "Base preparada", "La medición está disponible, pero no se implementó ni evaluó un generador adaptativo"],
        ["Aportar conocimiento para futuras investigaciones", "Cumplido", "Conjuntos revisados, cuadernos ejecutados, modelos guardados, métricas, límites y trabajo futuro"],
    ]
    story.append(academic_table(objective_table, [6.1*cm, 2.8*cm, 7.1*cm], styles))
    story.append(Spacer(1, 5))
    story.append(paragraph(
        "Resultado del Servicio Social: se entrega una plataforma experimental que puede volver a ejecutarse para observar señales oculares y hacer pruebas. No se entrega un sistema de manipulación ni una medida para clasificar psicológicamente al usuario.",
        styles,
        "note",
    ))

    story.append(paragraph("2. Introducción y descripción del problema técnico", styles, "h1"))
    story.append(paragraph(
        "La pupilometría mediante webcam es útil porque cuesta menos y resulta menos invasiva que un equipo especializado de seguimiento ocular. Sin embargo, un modelo sólo puede aprender correctamente si las etiquetas de entrenamiento representan lo que se quiere medir. El proyecto original trataba la localización del ojo, el centro pupilar, el diámetro, la mirada y la atención como si fueran lo mismo. BioID fue creado para detección facial y sólo proporciona posiciones generales de los ojos [1]; por eso no sirve para entrenar una elipse, un diámetro físico o un estado de atención.",
        styles,
    ))
    story.append(paragraph(
        "La implementación original aumentaba el problema porque recortaba la imagen alrededor de la misma anotación y colocaba la respuesta esperada cerca del centro del recorte. En una prueba con otros datos, las predicciones casi no cambiaban y el error era parecido al de responder siempre con un punto fijo. La pregunta de investigación se reformuló así: <b>¿puede un proceso ligero y local estimar el centro y el diámetro pupilar en personas que no aparecieron durante el entrenamiento, usando datos correctos para cada salida?</b>",
        styles,
    ))
    story.append(paragraph("2.1 Contribuciones", styles, "h2"))
    story.append(bullets([
        "Auditoría de correspondencia entre datos, etiquetas y afirmaciones de producto.",
        "Sustitución del modelo BioID por un modelo de centro entrenado con anotaciones LPW.",
        "Incorporación de un regresor compacto de diámetro basado en recortes webcam EyeDentify y referencia Tobii.",
        "Separación de participantes entre entrenamiento, validación y prueba, guardado de modelos y comparación contra soluciones sencillas.",
        "Prueba con elipses Cambridge y comprobación con BBBD/NEMAR de que el índice 0-100 actual no representa atención.",
    ], styles))
    story.append(paragraph(
        "Separar las tareas evita conclusiones incorrectas: localizar una pupila no demuestra su diámetro físico; medir la pupila no indica por sí solo dónde mira la persona; y ninguna de estas mediciones demuestra atención.",
        styles,
        "note",
    ))

    story.append(paragraph("3. Materiales y origen de los datos", styles, "h1"))
    dataset_table = [
        ["Fuente", "Objetivo", "Datos locales", "Separación y uso", "Licencia o tipo"],
        ["BioID", "Revisión del modelo anterior", "1,521 imágenes", "Sólo comparación", "Datos faciales; no pupila"],
        ["LPW", "Centro pupilar", "16,000 imágenes; 8 participantes", "6 entrenamiento / 1 validación / 1 prueba", "CC BY-NC-SA 4.0"],
        ["EyeDentify", "Diámetro en mm", "9,425 imágenes; 51 participantes", "36 entrenamiento / 8 validación / 7 prueba", "CC BY-NC 4.0"],
        ["Cambridge", "Elipse y diámetro en px", "300 imágenes; P1 y P2", "P1 ajuste / P2 prueba", "Uso académico con cita"],
        ["BBBD/NEMAR", "Condición atenta/distraída", "51 grabaciones; 27 sujetos", "24 pares completos", "CC BY 4.0"],
    ]
    story.append(academic_table(dataset_table, [2.4*cm, 3.4*cm, 3.2*cm, 3.4*cm, 3.3*cm], styles))
    story.append(Spacer(1, 5))
    story.append(figure(FIGURES / "figure_2_datasets.png", "Figura 2. Datos locales empleados. EyeDentify se separó por participante y cada fuente se usó para responder una pregunta específica.", styles, width=usable_width, max_height=7.1 * cm))

    story.append(paragraph("3.1 LPW", styles, "h2"))
    story.append(paragraph(
        "Labelled Pupils in the Wild contiene 66 vídeos de 22 participantes, registrados en ambientes cotidianos con variación interior/exterior, gafas, lentes de contacto y maquillaje [2]. Cada imagen incluye las coordenadas del centro. Se descargó un vídeo completo de 2,000 imágenes para cada uno de los participantes 1-8. Los archivos se compararon con el código MD5 publicado y se guardaron en archivos comprimidos de 64 x 64 px para acelerar el entrenamiento.", styles))
    story.append(paragraph("3.2 EyeDentify", styles, "h2"))
    story.append(paragraph(
        "EyeDentify relaciona recortes de webcam con mediciones Tobii y fue creado específicamente para estimar el diámetro usando equipo común [3]. El archivo público completo ocupa cerca de 34 GB. Se creó un lector ZIP64 que descarga sólo las partes necesarias del archivo: los datos de referencia y cuatro imágenes distribuidas en cada sesión de tres segundos. El subconjunto conserva los 51 participantes y las condiciones de iluminación disponibles.", styles))
    story.append(paragraph(
        "Los 51 participantes se ordenaron por su diámetro izquierdo promedio y se separaron en cuatro grupos de tamaño parecido. Con la semilla 42, se asignó 70% a entrenamiento, 15% a validación y el resto a prueba. Después se revisaron las 9,425 imágenes: no hubo archivos faltantes, dañados o duplicados, valores inválidos ni participantes repetidos entre grupos.", styles))

    story.append(paragraph("3.3 Muestras visuales de los participantes", styles, "h2"))
    story.append(paragraph(
        "La Figura 3 muestra ejemplos reales de las fuentes usadas. Los números indican participantes anónimos y no se incluyen nombres. BioID contiene el rostro completo y sólo marca posiciones generales de los ojos; por eso se conservó únicamente para explicar el problema del modelo anterior. LPW, EyeDentify y Cambridge trabajan con regiones del ojo y sí aportan información relacionada con la pupila.", styles))
    story.append(figure(
        FIGURES / "figure_3_sample_data.png",
        "Figura 3. Ejemplos de dos participantes o imágenes por fuente. La marca verde representa la información correcta disponible: puntos generales en BioID, centro pupilar en LPW y elipse en Cambridge. EyeDentify proporciona el diámetro Tobii mediante sus archivos de referencia.",
        styles,
        width=usable_width,
        max_height=8.0 * cm,
    ))

    story.append(paragraph("4. Métodos", styles, "h1"))
    story.append(paragraph("4.1 Modelo de centro pupilar", styles, "h2"))
    story.append(paragraph(
        "EyePupilCNN recibe una imagen monocromática de 64 x 64 px y regresa dos coordenadas normalizadas. La etiqueta original se transforma como x<sub>n</sub>=2x/ancho-1 y y<sub>n</sub>=2y/alto-1. El preprocesamiento convierte a gris, redimensiona mediante interpolación de área, aplica CLAHE (clipLimit 2.0, rejilla 8 x 8) y normaliza intensidades a [0,1].", styles))
    center_arch = [
        ["Etapa", "Configuración", "Salida aproximada"],
        ["Bloque 1", "Conv 3x3, 1->16; BN; ReLU; MaxPool 2", "16 x 32 x 32"],
        ["Bloque 2", "Conv 3x3, 16->32; BN; ReLU; MaxPool 2", "32 x 16 x 16"],
        ["Bloque 3", "Conv 3x3, 32->64; BN; ReLU; MaxPool 2", "64 x 8 x 8"],
        ["Regresor", "Flatten; Dense 128; ReLU; Dropout 0.30", "128"],
        ["Salida", "Dense 2; tanh", "(x, y) en [-1,1]"],
    ]
    story.append(academic_table(center_arch, [3.0*cm, 8.5*cm, 4.5*cm], styles))
    story.append(paragraph("4.2 Modelo de diámetro", styles, "h2"))
    story.append(paragraph(
        "PupilDiameterCNN recibe una imagen a color BGR de 64 x 32 px. El valor esperado del ojo izquierdo se normaliza usando sólo los datos de entrenamiento: media de 2.3836 mm y desviación de 0.2976 mm. Al usar el modelo se revierte esta normalización y, como medida de seguridad, la salida se limita a 1-9 mm. La red tiene 177,953 parámetros.", styles))
    diameter_arch = [
        ["Etapa", "Configuración"],
        ["Características", "Conv-BN-ReLU: 3->16->32->64->96; MaxPool tras los tres primeros bloques"],
        ["Agregación", "AdaptiveAvgPool a 2 x 4"],
        ["Regresor", "Dense 768->128; ReLU; Dropout 0.25; Dense 128->1"],
        ["Salida", "Diámetro estandarizado; desnormalización a milímetros"],
    ]
    story.append(academic_table(diameter_arch, [3.4*cm, 12.6*cm], styles))

    story.append(paragraph("4.3 Aumento y optimización", styles, "h2"))
    training_table = [
        ["Parámetro", "Centro LPW", "Diámetro EyeDentify"],
        ["Pérdida", "Smooth L1; beta=0.05", "L1 / MAE"],
        ["Optimizador", "AdamW; lr=0.001; wd=0.0002", "AdamW; lr=0.001; wd=0.0001"],
        ["Batch", "64", "64"],
        ["Máximo", "40 épocas", "50 épocas"],
        ["Scheduler", "ReduceLROnPlateau; factor 0.5; paciencia 3", "ReduceLROnPlateau; factor 0.5; paciencia 3"],
        ["Early stopping", "10 épocas sin mejora", "10 épocas sin mejora"],
        ["Aumentos", "Brillo/contraste; flip; traslación +/-4 px", "Brillo/contraste; blur; flip; ruido"],
        ["Mejor época", "12", "9"],
        ["Parada", "22", "23"],
    ]
    story.append(academic_table(training_table, [3.2*cm, 6.4*cm, 6.4*cm], styles))
    story.append(paragraph(
        "En LPW, la reflexión horizontal también invierte x y la traslación modifica ambas etiquetas. En EyeDentify se aplicó contraste 0.75-1.25 y brillo +/-18 con probabilidad 0.80, blur con 0.30, reflexión con 0.50 y ruido gaussiano de desviación 3 con 0.35. Todos los generadores se inicializaron con semilla 42.", styles))
    story.append(figure(FIGURES / "figure_3_learning.png", "Figura 4. Historia de entrenamiento y validación. El mejor modelo se eligió con el grupo de validación; el grupo de prueba no se usó hasta terminar esa selección.", styles, width=usable_width, max_height=7.0 * cm))

    story.append(paragraph("4.4 Procesamiento clásico y salidas temporales", styles, "h2"))
    story.append(paragraph(
        "El centro CNN define un recorte local de 32 x 32 px. Se aplica filtro gaussiano 5 x 5, umbral mínimo+35, cierre morfológico 3 x 3, selección del contorno mayor y ajuste de elipse por mínimos cuadrados con OpenCV [4]. El diámetro en píxeles es el promedio de los dos ejes. El offset 35 se seleccionó en Cambridge P1 y se verificó una sola vez en P2.", styles))
    story.append(paragraph(
        "AttentionTracker conserva su nombre para no romper la aplicación, pero no es un modelo de aprendizaje automático. Usa la mediana de 100 imágenes como valor de referencia, una ventana de 30 imágenes y factores fijos. Su salida se llama índice ocular experimental porque la pupila también cambia por la iluminación, el esfuerzo, la sorpresa y otras respuestas del cuerpo [5].", styles))

    story.append(paragraph("4.5 Recursos de cómputo", styles, "h2"))
    resource_table = [
        ["Recurso", "Configuración utilizada"],
        ["CPU", "Intel Core i5-11400H, 6 núcleos / 12 hilos, 2.70 GHz"],
        ["Memoria", "31.7 GiB RAM instalada"],
        ["Acelerador", "Ninguno; PyTorch CPU"],
        ["Software", "Python 3.14.2; PyTorch 2.12.1+cpu; OpenCV 4.13.0; NumPy 2.5.0"],
        ["Forma de ejecución", "Procesamiento local y sin Internet después de descargar los conjuntos de datos"],
    ]
    story.append(academic_table(resource_table, [4.0*cm, 12.0*cm], styles))

    story.append(paragraph("5. Protocolo de evaluación", styles, "h1"))
    story.append(paragraph(
        "La unidad independiente fue el participante, no cada imagen. Los mejores modelos se eligieron sólo con el grupo de validación. La prueba final reportó MAE, mediana del error absoluto, RMSE, percentil 95, sesgo, R², Pearson, Spearman y porcentaje dentro de tolerancias. Para el centro se usó la distancia directa entre dos puntos en el recorte de 64 x 64 px. Para el diámetro se comparó contra una solución sencilla que siempre responde con la media del entrenamiento.", styles))
    metric_table = [
        ["Métrica", "Interpretación"],
        ["MAE", "Promedio de |predicción - referencia|; error típico en unidades originales"],
        ["RMSE", "Penaliza con mayor fuerza los errores grandes"],
        ["P95 AE", "95% de los errores absolutos queda por debajo del valor"],
        ["Sesgo", "Promedio con signo; detecta sobreestimación o subestimación"],
        ["R²", "Varianza explicada respecto de predecir una constante"],
        ["Spearman", "Conservación del orden de diámetros sin exigir linealidad"],
        ["Wilcoxon", "Compara el MAE de cada participante contra la solución sencilla"],
    ]
    story.append(academic_table(metric_table, [3.4*cm, 12.6*cm], styles))
    story.append(paragraph(
        "La prueba geométrica externa se realizó en Cambridge P2: la CNN estimó el centro sin recibir la respuesta correcta y después OpenCV ajustó la elipse. Para revisar el índice de atención se aplicó la fórmula a BBBD/NEMAR y se comparó, mediante la prueba de Wilcoxon, la sesión atenta contra la distraída de cada participante.", styles))

    story.append(paragraph("6. Resultados", styles, "h1"))
    story.append(paragraph("6.1 Localización del centro", styles, "h2"))
    center_results = [
        ["Métrica en LPW P8 (n=2,000)", "BioID legado", "LPW nuevo", "Cambio"],
        ["Error medio", "10.374 px", "4.445 px", "-57.15%"],
        ["Mediana", "10.560 px", "4.244 px", "-59.81%"],
        ["P95", "16.257 px", "8.824 px", "-45.72%"],
        ["Dentro de 3 px", "3.70%", "31.30%", "+27.60 pp"],
        ["Dentro de 5 px", "7.80%", "62.65%", "+54.85 pp"],
    ]
    story.append(academic_table(center_results, [5.3*cm, 3.3*cm, 3.3*cm, 3.3*cm], styles))
    story.append(paragraph(
        "Las coordenadas del modelo anterior casi no cambiaban; la red LPW recuperó variación real (desviación x=0.188; desviación y=0.088). Esto muestra que cambiar los datos de entrenamiento corrigió el problema principal, aunque el error todavía es alto para calcular con precisión la mirada en pantalla.", styles))

    story.append(paragraph("6.2 Diámetro EyeDentify", styles, "h2"))
    diameter_results = [
        ["Métrica (n=1,280; 7 sujetos)", "CNN EyeDentify", "Media constante"],
        ["MAE", "0.202 mm", "0.224 mm"],
        ["Mediana AE", "0.185 mm", "0.210 mm"],
        ["RMSE", "0.244 mm", "0.260 mm"],
        ["P95 AE", "0.455 mm", "0.457 mm"],
        ["Dentro de +/-0.25 mm", "67.27%", "60.08%"],
        ["Dentro de +/-0.50 mm", "96.88%", "96.41%"],
        ["R²", "0.116", "-0.005"],
        ["Pearson / Spearman", "0.550 / 0.600", "No aplicable"],
        ["Sesgo", "+0.107 mm", "+0.019 mm"],
    ]
    story.append(academic_table(diameter_results, [6.4*cm, 4.8*cm, 4.8*cm], styles))
    story.append(paragraph(
        "El MAE se redujo 9.58%. El modelo fue mejor en seis de siete participantes, pero la prueba de Wilcoxon produjo p=0.296875. Para esta comparación cuentan siete personas independientes, no las 1,280 imágenes; por ello, el resultado todavía no permite asegurar que el modelo será mejor en la población general.", styles, "warning"))
    story.append(figure(FIGURES / "figure_4_results.png", "Figura 5. Comparación con soluciones sencillas. A la izquierda, cambiar BioID por LPW produce una mejora grande. A la derecha, EyeDentify mejora de forma moderada la respuesta constante, principalmente en MAE y RMSE.", styles, width=usable_width, max_height=7.0 * cm))
    story.append(figure(
        FIGURES / "figure_6_sample_predictions.png",
        "Figura 6. Ejemplos de prueba con datos que no se usaron para entrenar. Arriba se compara el centro correcto con el calculado. Abajo se muestran cuatro participantes EyeDentify con el diámetro Tobii y la salida del modelo.",
        styles,
        width=usable_width,
        max_height=7.5 * cm,
    ))

    story.append(paragraph("6.3 Prueba completa con otras imágenes", styles, "h2"))
    end_results = [
        ["Métrica Cambridge P2", "Resultado"],
        ["Imágenes", str(end_to_end["n"])],
        ["Tasa de detección", f"{end_to_end['detection_rate']:.0%}"],
        ["Error de centro CNN", f"{end_to_end['cnn_center_mae_px']:.3f} px"],
        ["Error de centro tras elipse", f"{end_to_end['ellipse_center_mae_px']:.3f} px"],
        ["MAE de diámetro", f"{end_to_end['diameter_mae_px']:.3f} px"],
        ["Mediana / P95", f"{end_to_end['diameter_median_ae_px']:.3f} / {end_to_end['diameter_p95_ae_px']:.3f} px"],
        ["Sesgo", f"{end_to_end['diameter_bias_px']:+.3f} px"],
        ["Spearman", f"{end_to_end['diameter_spearman']:.3f}"],
    ]
    story.append(academic_table(end_results, [8.0*cm, 8.0*cm], styles))
    story.append(Spacer(1, 5))
    story.append(figure(FIGURES / "figure_5_end_to_end.png", "Figura 7. Prueba completa sin entregar al algoritmo el centro correcto. Las estimaciones siguen bien el orden de los valores reales, aunque tienden a ser menores y algunos errores llegan hasta un P95 de 2.131 px.", styles, width=usable_width, max_height=7.1 * cm))
    story.append(paragraph(
        "La elipse redujo el error de centro de 5.708 a 1.764 px. Esto muestra que la CNN encuentra una zona aproximada y el método clásico mejora la precisión. El sesgo de -0.681 px indica que el diámetro suele estimarse por debajo del valor real. En una prueba de sensibilidad, oscurecer la imagen 40 niveles elevó el MAE a 6.992 px; por eso la iluminación es el principal riesgo técnico.", styles))

    story.append(paragraph("6.4 Prueba del índice ocular", styles, "h2"))
    story.append(paragraph(
        "En 24 participantes con ambas sesiones, el índice promedio fue 80.734 en la condición atenta y 86.616 en la distraída. La diferencia atenta-distraída fue -5.882 (p=0.000430) y sólo 5 de 24 participantes siguieron la dirección esperada. También se probó un clasificador dejando fuera del entrenamiento a cada participante de prueba. Obtuvo AUC de 0.696 y exactitud balanceada de 0.641. Esto indica que las señales contienen algo de información, pero la fórmula fija las combina de forma incorrecta.", styles))
    story.append(figure(RESEARCH / "attention_construct_validation.png", "Figura 8. Comparación de atención por participante. La mayoría de las líneas aumenta en la condición distraída; por lo tanto, la fórmula 0-100 no puede leerse como una probabilidad de atención.", styles, width=usable_width, max_height=7.1 * cm))

    story.append(paragraph("7. Revisión de criterios y viabilidad", styles, "h1"))
    viability = [
        ["Criterio original", "Estado", "Evidencia"],
        ["SC-001: varianza <0.8 px en reposo", "No demostrado", "Cambridge mide exactitud estática (MAE 0.876 px), no repetibilidad en reposo."],
        ["SC-002: sistema >=15 FPS en CPU", "Parcial", "Centro 1.681 ms/ojo y diámetro 0.640 ms/ojo; falta medir en conjunto la captura, detección facial, OpenCV e interfaz."],
        ["SC-003: atención reacciona <1 s", "No válido", "La fórmula se actualiza, pero entrega el resultado contrario al esperado en datos pareados."],
        ["SC-004: reporte <2 s", "Cumplido", f"600 imágenes exportadas en {reporter['generation_seconds']:.3f} s; archivos Markdown y PNG verificados."],
    ]
    story.append(academic_table(viability, [4.7*cm, 2.6*cm, 8.7*cm], styles))
    story.append(paragraph("7.1 Viabilidad por componente", styles, "h2"))
    story.append(bullets([
        "<b>Centro pupilar:</b> útil para recortar y mejorar la zona de la pupila; todavía no basta para calcular la mirada en pantalla sin calibración y postura de cabeza.",
        "<b>Diámetro en píxeles:</b> prometedor como señal relativa bajo iluminación controlada; vulnerable a cambios tonales.",
        "<b>Diámetro en milímetros:</b> viable para investigación y calibración por dispositivo; no clínico, con R²=0.116 y evidencia por sujeto limitada.",
        "<b>Tiempo real:</b> las CNN ofrecen amplio margen; la cifra de FPS integral debe medirse en la webcam objetivo.",
        "<b>Atención:</b> no viable con la fórmula actual. La interfaz sólo muestra un índice ocular experimental.",
        "<b>Uso comercial:</b> limitado por las licencias no comerciales de LPW y EyeDentify salvo autorización adicional.",
    ], styles))

    story.append(paragraph("8. Discusión", styles, "h1"))
    story.append(paragraph(
        "El resultado principal no se debe a una red más complicada, sino a usar etiquetas que sí representan la variable buscada. Una red pequeña con datos correctos superó ampliamente al modelo entrenado con imágenes que no describían el centro pupilar. La reducción de 57.15% en el error de centro apoya esta explicación. En diámetro, la diferencia contra predecir siempre la media es menor: la CNN conserva el orden general y obtiene R² positivo, pero también presenta un sesgo de +0.107 mm y diferencias importantes entre personas.", styles))
    story.append(paragraph(
        "LPW incluye muchas posiciones y condiciones, pero proviene de un equipo infrarrojo colocado en la cabeza. EyeDentify se parece más al uso con webcam, aunque sus recortes tienen baja resolución y el subconjunto sólo usa cuatro momentos por sesión. El modelo de diámetro se entrenó con el ojo izquierdo. Voltear horizontalmente las imágenes ayuda a que el modelo responda ante más variaciones, pero no reemplaza una prueba separada con el ojo derecho. Además, el tamaño observado depende de la distancia y de la cámara, factores que el modelo puede relacionar por error con cada persona.", styles))
    story.append(paragraph(
        "La prueba Cambridge usó datos diferentes para elegir y evaluar el umbral. Su correlación de 0.785 muestra que el método responde a cambios reales. Sin embargo, 150 imágenes de una sola persona no forman un grupo suficiente para una evaluación clínica. De la misma forma, 1,280 imágenes EyeDentify no equivalen a 1,280 casos independientes: la comparación estadística corresponde a siete participantes.", styles))
    story.append(paragraph(
        "La pupilometría cambia por varios procesos y también por la iluminación [5]. El resultado BBBD/NEMAR muestra que no basta con comprobar que una fórmula funciona sin errores: también debe representar lo que dice medir. Una fórmula puede ser rápida y estable, pero entregar un valor con un significado incorrecto.", styles))

    story.append(paragraph("9. Limitaciones del estudio", styles, "h1"))
    threats = [
        ["Tipo", "Limitación", "Cómo se atendió / trabajo pendiente"],
        ["Datos", "Las imágenes de una misma sesión se parecen entre sí", "Separación por participante; calcular intervalos futuros por persona"],
        ["Tipo de imagen", "LPW infrarrojo frente a webcam RGB", "Prueba externa Cambridge; falta un grupo local con webcam"],
        ["Cámara", "Iluminación, distancia y escala de cámara", "Registrar lux/distancia y calibrar por dispositivo"],
        ["Ojos", "Ojo derecho no evaluado directamente", "Crear grupos de prueba para ambos ojos y reportar cada uno"],
        ["Estadística", "Sólo siete participantes de prueba en diámetro", "Ampliar participantes y calcular intervalos de confianza"],
        ["Atención", "Pupila y estabilidad no equivalen a atención", "Usar tareas, desempeño, preguntas durante la prueba e iluminación controlada"],
        ["Velocidad", "FPS medido por modelo y no en todo el sistema", "Medir de forma continua cámara, detección e interfaz"],
    ]
    story.append(academic_table(threats, [2.4*cm, 6.0*cm, 7.6*cm], styles))

    story.append(paragraph("10. Archivos entregados y repetición de las pruebas", styles, "h1"))
    story.append(paragraph(
        "El repositorio conserva el programa de descarga con comprobaciones, la lista exacta de datos usados, las curvas por época, las predicciones por imagen, las métricas por participante, los cuadernos ejecutados y los códigos SHA-256 de los modelos guardados. Los datos y modelos no se suben a Git por su tamaño y licencia, pero permanecen en el equipo local.", styles))
    reproducibility = [
        ["Archivo o resultado", "Ubicación"],
        ["Lista de datos EyeDentify", "docs/training/dataset_manifest_snapshot.csv"],
        ["Resultados por época", "docs/training/center_history.csv; history.csv"],
        ["Predicciones finales", "docs/training/center_test_predictions.csv; test_predictions.csv"],
        ["Prueba Cambridge", "docs/training/end_to_end_predictions.csv"],
        ["Resumen JSON", "docs/training/final_validation_summary.json"],
        ["Cuaderno final", "notebooks/04_final_training_and_validation.ipynb"],
        ["Modelos activos", "src/models/eyestim_cnn.pth; eyestim_diameter.pth"],
    ]
    story.append(academic_table(reproducibility, [5.2*cm, 10.8*cm], styles))

    story.append(paragraph("11. Conclusiones del Servicio Social", styles, "h1"))
    story.append(paragraph(
        "Se corrigió el problema principal entre el propósito de EyeStim y los datos originales. LPW permitió que el modelo dejara de predecir casi siempre el mismo punto y ahora sirve para guiar el ajuste de la elipse. EyeDentify permitió estimar el diámetro en milímetros usando medidas Tobii como referencia. Cambridge confirmó que el proceso completo también funciona con imágenes diferentes a las de entrenamiento. Los experimentos pueden volver a ejecutarse y los modelos quedaron integrados en la interfaz.", styles))
    story.append(paragraph(
        "La conclusión debe ser clara y limitada: EyeStim es un <b>prototipo de pupilometría experimental</b>. La medición absoluta requiere calibración local y más participantes; el punto de la pupila no indica directamente la mirada en pantalla; y el índice 0-100 no mide atención. Reconocer estos límites permite indicar qué resultados están comprobados y qué pruebas faltan.", styles, "note"))
    story.append(paragraph(
        "Respecto de la propuesta de Servicio Social, se completó el sistema necesario para observar y registrar la respuesta ocular, se entrenaron modelos propios y se revisó su funcionamiento con varias pruebas. La generación de estímulos audiovisuales adaptativos queda como trabajo futuro. Antes se necesita una medida de atención válida y un protocolo con consentimiento, tareas con resultados observables y control de iluminación. Crear esa función sin estas pruebas podría producir conclusiones equivocadas.", styles))

    story.append(PageBreak())
    story.append(paragraph("Referencias", styles, "h1"))
    references = [
        "[1] BioID GmbH. <i>BioID Face Database</i>. https://www.bioid.com/face-database/",
        "[2] Tonsen, M., Zhang, X., Sugano, Y. y Bulling, A. (2016). <i>Labelled Pupils in the Wild: A Dataset for Studying Pupil Detection in Unconstrained Environments</i>. ETRA. Dataset: https://doi.org/10.18419/DARUS-3237",
        "[3] Shah, V. et al. (2024). <i>EyeDentify: A Dataset for Pupil Diameter Estimation based on Webcam Images</i>. arXiv:2407.11204. https://github.com/vijulshah/eyedentify",
        "[4] Bradski, G. (2000). <i>The OpenCV Library</i>. Dr. Dobb's Journal of Software Tools.",
        "[5] Mathôt, S. (2018). <i>Pupillometry: Psychology, Physiology, and Function</i>. Psychonomic Bulletin & Review, 25, 1694-1708. https://doi.org/10.3758/s13423-018-1432-y",
        "[6] Świrski, L., Bulling, A. y Dodgson, N. (2012). <i>Robust Real-Time Pupil Tracking in Highly Off-Axis Images</i>. ETRA. https://www.cl.cam.ac.uk/research/rainbow/projects/pupiltracking/datasets/",
        "[7] Paszke, A. et al. (2019). <i>PyTorch: An Imperative Style, High-Performance Deep Learning Library</i>. NeurIPS 32.",
        "[8] Loshchilov, I. y Hutter, F. (2019). <i>Decoupled Weight Decay Regularization</i>. ICLR.",
        "[9] NEMAR. <i>Dataset nm000150: Brain, Body, and Behavior Dataset</i>. https://www.nemar.org/dataset/nm000150",
        "[10] Wilcoxon, F. (1945). <i>Individual Comparisons by Ranking Methods</i>. Biometrics Bulletin, 1(6), 80-83.",
    ]
    for reference in references:
        story.append(Paragraph(reference, styles["reference"]))

    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=0.7, color=RULE, spaceBefore=4, spaceAfter=5))
    story.append(paragraph(
        "Nota final: todos los resultados provienen de pruebas realizadas localmente con el repositorio. No se usaron servicios comerciales de estimación ocular ni GPU. Los conjuntos de datos conservan sus licencias y referencias.", styles, "caption"))

    document.build(story, onFirstPage=page_decor, onLaterPages=page_decor)
    return OUTPUT


if __name__ == "__main__":
    print(build())
