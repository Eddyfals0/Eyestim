import os
import time
import numpy as np
import matplotlib.pyplot as plt
import config

class SessionReporter:
    """
    Registra datos temporales del tracking de mirada y pupilometría,
    y genera informes estructurados con Matplotlib y Markdown.
    """
    def __init__(self, output_dir: str = config.DOCS_DIR):
        self.output_dir = output_dir
        self.timestamps = []
        self.pupil_diameters = []
        self.gaze_coords_x = []
        self.gaze_coords_y = []
        self.attention_scores = []
        self.active_zones = []
        
        os.makedirs(self.output_dir, exist_ok=True)

    def log_frame(self, diameter: float, nx: float, ny: float, attention: float, zone: str) -> None:
        """
        Registra un fotograma de la sesión.
        """
        self.timestamps.append(time.time())
        self.pupil_diameters.append(diameter)
        self.gaze_coords_x.append(nx)
        self.gaze_coords_y.append(ny)
        self.attention_scores.append(attention)
        self.active_zones.append(zone)

    def generate_report(self, baseline_diameter: float = 0.0) -> tuple[str, str]:
        """
        Calcula estadísticas, genera gráficos temporales en PNG
        y escribe el reporte Markdown.
        
        Returns:
            report_path (str): Ruta al archivo docs/session_report.md.
            graph_path (str): Ruta al archivo docs/attention_evolution.png.
        """
        total_frames = len(self.timestamps)
        if total_frames == 0:
            print("[Reporter] No hay datos registrados en la sesión. Omitiendo reporte.")
            return "", ""

        # 1. Calcular duración y promedios
        duration_sec = self.timestamps[-1] - self.timestamps[0]
        avg_attention = np.mean(self.attention_scores)
        valid_diameters = [d for d in self.pupil_diameters if d > 0.0]
        avg_diameter = np.mean(valid_diameters) if valid_diameters else 0.0
        
        # 2. Calcular estadísticas por zona
        zones, counts = np.unique(self.active_zones, return_counts=True)
        zone_distribution = {z: 0.0 for z in ["Centro", "Arriba", "Abajo", "Izquierda", "Derecha"]}
        for z, c in zip(zones, counts):
            if z in zone_distribution:
                zone_distribution[z] = (c / total_frames) * 100.0

        # 3. Generar gráficos temporales con Matplotlib
        rel_time = np.array(self.timestamps) - self.timestamps[0]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Subplot 1: Evolución del diámetro de la pupila
        ax1.plot(rel_time, self.pupil_diameters, color="blue", linewidth=1.5, label="Diámetro actual")
        if baseline_diameter > 0.0:
            ax1.axhline(y=baseline_diameter, color="red", linestyle="--", label=f"Línea base ({baseline_diameter:.1f} px)")
        ax1.set_ylabel("Diámetro Pupilar (píxeles)")
        ax1.set_title("Evolución de la Dilatación Pupilar y Atención del Usuario")
        ax1.legend(loc="upper right")
        ax1.grid(True)
        
        # Subplot 2: Evolución de la atención
        ax2.plot(rel_time, self.attention_scores, color="green", linewidth=1.5, label="Porcentaje de atención")
        ax2.axhline(y=70, color="orange", linestyle=":", label="Umbral de alta concentración (70%)")
        ax2.set_xlabel("Tiempo transcurrido (segundos)")
        ax2.set_ylabel("Porcentaje de Atención (%)")
        ax2.set_ylim(-5, 105)
        ax2.legend(loc="upper right")
        ax2.grid(True)
        
        plt.tight_layout()
        graph_path = os.path.join(self.output_dir, "attention_evolution.png")
        plt.savefig(graph_path)
        plt.close()
        
        # 4. Crear reporte en Markdown
        report_path = os.path.join(self.output_dir, "session_report.md")
        
        # Formatear la tabla de permanencia por zona
        table_rows = []
        for zone_name, pct in zone_distribution.items():
            count_frames = self.active_zones.count(zone_name)
            table_rows.append(f"| **{zone_name}** | {count_frames} | {pct:.2f}% |")
        table_content = "\n".join(table_rows)
        
        markdown_content = f"""# Reporte Analítico de Atención y Pupilometría
**Fecha de Análisis**: {time.strftime('%Y-%m-%d %H:%M:%S')}
**Duración de la Sesión**: {duration_sec:.2f} segundos
**Muestras totales (Fotogramas)**: {total_frames}

## Resumen Ejecutivo

- **Porcentaje de Atención Promedio**: {avg_attention:.2f}%
- **Diámetro Pupilar Promedio**: {avg_diameter:.2f} píxeles
- **Línea Base del Sujeto**: {baseline_diameter:.2f} píxeles (calibrada)

---

## Distribución de Atención por Zonas

La siguiente tabla detalla la cantidad de fotogramas y el porcentaje de tiempo que el usuario permaneció enfocado en cada región de la pantalla:

| Zona de Enfoque | Fotogramas (Muestras) | Porcentaje de Permanencia |
| :--- | :---: | :---: |
{table_content}

---

## Gráficos de Evolución Temporal

La evolución en tiempo real de las variables cognitivas y fisiológicas registradas durante la sesión se detalla a continuación:

![Evolución Temporal de la Atención](attention_evolution.png)

---

## Notas de Interpretación Fisiológica

1. **Dilatación Cognitiva**: Variaciones positivas en el diámetro de la pupila por encima de la línea base calibrada ({baseline_diameter:.1f} px), sin cambios en la iluminación ambiental, indican un incremento de la carga cognitiva o atención concentrada del sujeto.
2. **Estabilidad de la Mirada**: Una varianza baja en la zona de enfoque se correlaciona con periodos de lectura fija o atención sostenida, empujando al alza el porcentaje de atención calculado.
3. **Parpadeos**: Los valles de caída a 0% de atención representan parpadeos o oclusiones momentáneas del rostro del usuario, lo cual es normal y saludable durante la visualización prolongada.
"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
            
        print(f"[Reporter] Reporte Markdown creado en: {report_path}")
        print(f"[Reporter] Gráfico analítico exportado a: {graph_path}")
        return report_path, graph_path

if __name__ == "__main__":
    # Test unitario autónomo
    print("=== Test Unitario Autónomo del Generador de Reportes ===")
    reporter = SessionReporter()
    
    # Simular una sesión de 5 segundos a 10 FPS (50 fotogramas)
    start_t = time.time()
    for i in range(50):
        # Simular algunos parpadeos (caída a 0)
        d = 0.0 if i in [20, 21] else (10.0 + np.random.uniform(-0.3, 0.5))
        att = 0.0 if d == 0.0 else (80.0 + np.random.uniform(-5.0, 10.0))
        z = "Centro" if i < 35 else "Derecha"
        
        reporter.log_frame(d, 0.0, 0.0, att, z)
        # Ajustar timestamps de prueba
        reporter.timestamps[i] = start_t + (i * 0.1)
        
    rep_p, gr_p = reporter.generate_report(baseline_diameter=10.0)
    
    assert os.path.exists(rep_p), "Error: No se creó el reporte Markdown."
    assert os.path.exists(gr_p), "Error: No se creó el gráfico PNG."
    print("Módulo de reporte verificado correctamente en aislamiento.")
