import os
import csv
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox
from analyze_iris import find_iris_center, get_images
from model import DetectorDeOjosCNN
# Asegurarse que pillow soporta dibujar para la cruz
from PIL import ImageDraw

class ValidatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Validador de Ojos y Localizador de Iris")
        self.root.geometry("600x650")

        # Estado
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.ruta_datos = self.get_data_path()
        self.image_paths = get_images(self.ruta_datos, num_images=1000) # Cargar muchas para tener de sobra
        self.current_idx = 0
        self.feedback_file = os.path.join(self.base_dir, "data", "feedback_labels.csv")
        self.prediccion_actual = None # "ojo" o "sin_ojo"
        
        # Red Neuronal
        self.dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.modelo = DetectorDeOjosCNN()
        self.modelo_cargado = self.load_model()
        
        self.transformacion_modelo = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.to_tensor_simple = transforms.ToTensor()

        # Interfaz de Usuario
        self.setup_ui()
        
        # Iniciar si hay imágenes
        if not self.image_paths:
            messagebox.showerror("Error", f"No se encontraron imágenes en {self.ruta_datos}")
        else:
            self.load_current_image()

    def get_data_path(self):
        ruta = os.path.join(self.base_dir, "data", "train", "forward_look")
        if not os.path.exists(ruta):
            ruta = os.path.join(self.base_dir, "..", "data", "train", "forward_look")
        return ruta

    def load_model(self):
        ruta_modelo = os.path.join(self.base_dir, "models", "mi_modelo_detectar_ojos.pth")
        if not os.path.exists(ruta_modelo):
            ruta_modelo = os.path.join(self.base_dir, "..", "models", "mi_modelo_detectar_ojos.pth")
            
        if os.path.exists(ruta_modelo):
            self.modelo.load_state_dict(torch.load(ruta_modelo, map_location=self.dispositivo))
            self.modelo.to(self.dispositivo)
            self.modelo.eval()
            return True
        return False

    def setup_ui(self):
        # Marco Principal
        main_frame = tk.Frame(self.root, padx=20, pady=20)
        main_frame.pack(expand=True, fill="both")

        # Status Label Superior
        self.status_lbl = tk.Label(main_frame, text="Estado: Listo", font=("Arial", 12))
        self.status_lbl.pack(pady=5)

        # Imagen (Canvas)
        self.canvas = tk.Canvas(main_frame, width=256, height=256, bg="gray")
        self.canvas.pack(pady=10)
        
        # Botones de Acción
        action_frame = tk.Frame(main_frame)
        action_frame.pack(pady=10)
        
        tk.Button(action_frame, text="🔍 Analizar Imagen", command=self.analyze_image, 
                  font=("Arial", 12, "bold"), bg="lightblue").pack(side=tk.LEFT, padx=10)

        # Botones de Feedback
        feedback_frame = tk.Frame(main_frame)
        feedback_frame.pack(pady=10)
        
        self.btn_good = tk.Button(feedback_frame, text="✅ Bien (Correcto)", command=lambda: self.save_feedback(True),
                                  font=("Arial", 11), bg="lightgreen", state=tk.DISABLED)
        self.btn_good.pack(side=tk.LEFT, padx=10)
        
        self.btn_bad = tk.Button(feedback_frame, text="❌ Mal (Incorrecto)", command=lambda: self.save_feedback(False),
                                 font=("Arial", 11), bg="salmon", state=tk.DISABLED)
        self.btn_bad.pack(side=tk.LEFT, padx=10)

        # Controles de Navegación
        nav_frame = tk.Frame(main_frame)
        nav_frame.pack(pady=20)
        
        tk.Button(nav_frame, text="⬅️ Anterior", command=self.prev_image, width=10).pack(side=tk.LEFT, padx=10)
        self.lbl_counter = tk.Label(nav_frame, text="0 / 0", font=("Arial", 10))
        self.lbl_counter.pack(side=tk.LEFT, padx=10)
        tk.Button(nav_frame, text="Siguiente ➡️", command=self.next_image, width=10).pack(side=tk.LEFT, padx=10)
        
        # Re-Entrenar
        tk.Button(main_frame, text="🔄 Re-entrenar Red Neuronal (!)", command=self.trigger_retrain,
                  font=("Arial", 10, "italic")).pack(pady=20)

    def load_current_image(self):
        if not self.image_paths:
            return
            
        # Resetear estado
        self.btn_good.config(state=tk.DISABLED)
        self.btn_bad.config(state=tk.DISABLED)
        self.status_lbl.config(text="Estado: Esperando Análisis", fg="black")
        self.prediccion_actual = None
        
        # Mostrar foto original
        ruta_img = self.image_paths[self.current_idx]
        img = Image.open(ruta_img).convert('RGB')
        
        # Escalar para que se vea claro
        img = img.resize((256, 256), Image.Resampling.LANCZOS)
        
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(128, 128, image=self.photo)
        self.lbl_counter.config(text=f"{self.current_idx + 1} / {len(self.image_paths)}")

    def next_image(self):
        if self.current_idx < len(self.image_paths) - 1:
            self.current_idx += 1
            self.load_current_image()

    def prev_image(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_current_image()

    def analyze_image(self):
        ruta_img = self.image_paths[self.current_idx]
        img_cruda = Image.open(ruta_img).convert('RGB')
        
        # 1. Localizar Iris (Gaussian Blur -> min pixel)
        img_tensor = self.to_tensor_simple(img_cruda)
        x_iris, y_iris = find_iris_center(img_tensor)
        
        # Dibujar resultado del iris sobre la imagen (Redimensionando coordenadas)
        escala = 256 / 64  # Asumiendo imagen original de 64x64
        x_render = x_iris * escala
        y_render = y_iris * escala
        
        # Dibujar cruz roja
        img_dibujo = img_cruda.copy().resize((256, 256), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(img_dibujo)
        r = 10 # tamaño cruz
        draw.line((x_render-r, y_render, x_render+r, y_render), fill="red", width=3)
        draw.line((x_render, y_render-r, x_render, y_render+r), fill="red", width=3)
        
        self.photo = ImageTk.PhotoImage(img_dibujo)
        self.canvas.create_image(128, 128, image=self.photo)

        # 2. Evaluar Red Neuronal
        if self.modelo_cargado:
            img_modelo = self.transformacion_modelo(img_cruda).unsqueeze(0).to(self.dispositivo)
            with torch.no_grad():
                salida = self.modelo(img_modelo)
                porcentaje = torch.sigmoid(salida).item()
                
            if porcentaje < 0.5:
                certeza = (1 - porcentaje) * 100
                self.prediccion_actual = "ojo" # Clase 0
                self.status_lbl.config(text=f"Predicción: OJO ({certeza:.1f}%)", fg="green")
            else:
                certeza = porcentaje * 100
                self.prediccion_actual = "sin_ojo" # Clase 1
                self.status_lbl.config(text=f"Predicción: NO OJO ({certeza:.1f}%)", fg="red")
                
            # Activar botones
            self.btn_good.config(state=tk.NORMAL)
            self.btn_bad.config(state=tk.NORMAL)
        else:
            self.status_lbl.config(text="Iris localizado (Red Neuronal no encontrada)", fg="orange")

    def save_feedback(self, is_correct):
        ruta_img = self.image_paths[self.current_idx]
        
        # Si fue correcta, la etiqueta real es la predicción.
        # Si fue mala, la etiqueta es la contraria.
        if is_correct:
            etiqueta_real = self.prediccion_actual
        else:
            etiqueta_real = "sin_ojo" if self.prediccion_actual == "ojo" else "ojo"
            
        # Crear carpeta de datos si no existe
        os.makedirs(os.path.dirname(self.feedback_file), exist_ok=True)
        
        # Guardar en CSV
        archivo_existe = os.path.exists(self.feedback_file)
        with open(self.feedback_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not archivo_existe:
                writer.writerow(["RutaImagen", "EtiquetaReal"]) # Cabeceras
            writer.writerow([ruta_img, etiqueta_real])
            
        print(f"📝 Registrado feedback para {os.path.basename(ruta_img)} -> {etiqueta_real}")
        self.status_lbl.config(text=f"✅ Feedback guardado! (Etiqueta: {etiqueta_real})", fg="blue")
        
        # Desactivar botones para evitar dobles clicks
        self.btn_good.config(state=tk.DISABLED)
        self.btn_bad.config(state=tk.DISABLED)
        
        # Avanzar tras medio segundo
        self.root.after(600, self.next_image)

    def trigger_retrain(self):
        if not os.path.exists(self.feedback_file):
            messagebox.showinfo("Sin Datos", "No hay feedback guardado aún. Clasifica algunas imágenes como Bien/Mal primero.")
            return
            
        respuesta = messagebox.askyesno("Re-Entrenar", "¿Deseas re-entrenar la red neuronal usando el feedback recolectado?\n\nEsto actualizará el modelo guardado.")
        if respuesta:
            self.status_lbl.config(text="Entrenando... Revisa la consola", fg="orange")
            self.root.update()
            
            # Importar e invocar la función de reentrenamiento aquí
            import train
            exito = train.reentrenar(self.feedback_file)
            
            if exito:
                self.status_lbl.config(text="¡Re-entrenamiento exitoso!", fg="green")
                # Recargar el nuevo modelo
                self.load_model()
            else:
                self.status_lbl.config(text="Fallo al re-entrenar", fg="red")

if __name__ == "__main__":
    root = tk.Tk()
    app = ValidatorApp(root)
    root.mainloop()
