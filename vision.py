# ====== IMPORTS ======
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import easyocr
import numpy as np
import re
import torch

class ModuloVision:
    
    # ====== INICIALIZACIÓN DE HARDWARE Y MODELOS ======
    # 1. Configurar el entorno de ejecución (GPU/CPU) y carga los modelos de IA en memoria
    def __init__(self):
        # 2. Detectamos automáticamente si hay una GPU NVIDIA disponible para acelerar el proceso
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"--- Sistema de Visión iniciando en: {self.device.upper()} ---")

        print("Cargando modelo BLIP localmente (Vía clases nativas)...")
        # Cargamos el procesador de imágenes de Hugging Face
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        # 3. Enviamos el modelo BLIP a la memoria de la GPU (.to(device)) para mayor velocidad
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
        print("¡Modelo BLIP cargado!")

        print("Cargando lector OCR...")
        # 4. Enlazamos EasyOCR a tu GPU dinámicamente si está disponible
        usar_gpu = True if self.device == "cuda" else False
        self.lector_ocr = easyocr.Reader(['en'], gpu=usar_gpu) 
        print("¡Lector OCR listo para analizar!")

    # ====== LIMPIEZA DE DATOS OCR ======
    def extraer_numeros(self, texto_leido):
        """
        Limpia el texto detectado para quedarse solo con los números.
        Ahora ignora comas y puntos para scores grandes (ej. '35,322' -> '35322').
        """
        if not texto_leido:
            return "0"
        
        # Unimos todo y borramos las comas o puntos que confunden al OCR
        texto_unido = " ".join(texto_leido).replace(",", "").replace(".", "")
        # Extraemos mediante expresiones regulares solo los dígitos numéricos
        numeros = re.findall(r'\d+', texto_unido)
        return numeros[0] if numeros else "0"

    # ====== ANÁLISIS PRINCIPAL DEL FRAME ======
    # Toma una imagen del juego, describe qué está pasando e identifica estadísticas clave
    def clasificar_imagen(self, ruta_imagen):
        print(f"Analizando {ruta_imagen}...")
        
        try:
            # Abre la imagen y asegura el formato de color correcto para evitar errores con los modelos
            imagen = Image.open(ruta_imagen).convert('RGB')
            
            # --- PARTE A: LA "VIBRA" CON BLIP (ANÁLISIS DE CONTEXTO) ---
            texto_guia = "a game of Tetris showing"
            
            # 5. ¡IMPORTANTE! Las imágenes también deben enviarse a la GPU para ser procesadas
            inputs = self.processor(imagen, text=texto_guia, return_tensors="pt").to(self.device)
            
            # Genera la descripción textual de lo que la IA ve en la pantalla
            out = self.model.generate(
                **inputs, 
                max_new_tokens=40,
                repetition_penalty=1.5 
            )
            descripcion = self.processor.decode(out[0], skip_special_tokens=True)
            
            # --- PARTE B: LECTURA DE DATOS CON OCR (RECONOCIMIENTO DE TEXTO) ---
            
            # 1. Recorte del SPEED LV (Bajamos de 740->790 a 760->810)
            coordenadas_speed = (650, 790, 705, 832) 
            recorte_speed = imagen.crop(coordenadas_speed)
            #recorte_speed.show()
            
            texto_speed_crudo = self.lector_ocr.readtext(np.array(recorte_speed), detail=0)
            speed_final = self.extraer_numeros(texto_speed_crudo)

            # 2. Recorte de las LINES (Bajamos de 840->890 a 860->910)
            coordenadas_lines = (640, 893, 707, 942) 
            recorte_lines = imagen.crop(coordenadas_lines)
            #recorte_lines.show()
            
            texto_lines_crudo = self.lector_ocr.readtext(np.array(recorte_lines), detail=0)
            lines_final = self.extraer_numeros(texto_lines_crudo)

            # 3. Recorte del SCORE (Bajamos de 750->810 a 770->830)
            coordenadas_score = (1227, 785, 1425, 838) 
            recorte_score = imagen.crop(coordenadas_score)
            #recorte_score.show()
            
            texto_score_crudo = self.lector_ocr.readtext(np.array(recorte_score), detail=0)
            score_final = self.extraer_numeros(texto_score_crudo)

            # 4. NUEVO: Recorte de EVENTOS ESPECIALES (Lado izquierdo, zona central)
            coordenadas_evento = (436, 312, 698, 570) 
            recorte_evento = imagen.crop(coordenadas_evento)
            #recorte_evento.show() # Quita el '#' si necesitas ajustar esta nueva caja
            
            texto_evento_crudo = self.lector_ocr.readtext(np.array(recorte_evento), detail=0)
            # Unimos todo el texto y lo pasamos a mayúsculas para buscar coincidencias exactas
            texto_evento_unido = " ".join(texto_evento_crudo).upper()
            
            # Lógica para detectar jugadas críticas basadas en el texto emergente del juego
            evento_detectado = None
            if "TETR" in texto_evento_unido: # Buscamos solo "TETR" por si la 'I' o 'S' fallan en el OCR
                evento_detectado = "TETRIS"
            elif "SPIN" in texto_evento_unido:
                evento_detectado = "T-SPIN"
                
            es_back_to_back = "BACK" in texto_evento_unido

            # ====== EMPAQUETADO DE RESULTADOS ======
            # 5. Devolvemos el paquete maestro con toda la información extraída estructurada en un diccionario
            return {
                "descripcion": descripcion,
                "speed_lv": int(speed_final),
                "lines": int(lines_final),
                "score": int(score_final),
                "evento": evento_detectado,
                "back_to_back": es_back_to_back
            }
                
        # ====== MANEJO DE ERRORES ======
        except FileNotFoundError:
            print(f"Error: No se encontró la imagen en la ruta '{ruta_imagen}'")
            return None
        except Exception as e:
            print(f"Error procesando la imagen: {e}")
            return None

# =========================================================
# PRUEBA DEL MÓDULO (MAIN)
# =========================================================
if __name__ == "__main__":
    # Instanciamos la clase de visión
    vision = ModuloVision()
    
    # Cambia esto por el nombre exacto de la foto donde sale el T-Spin
    imagen_prueba = "frames_capturados/frame_0085.jpg" 
    
    # Ejecutamos el análisis de la imagen de prueba
    resultado = vision.clasificar_imagen(imagen_prueba)
    
    # Imprimimos los resultados en consola para verificar que las coordenadas y la IA funcionan bien
    if resultado:
        print("\n--- RESULTADO DE LA VISIÓN HÍBRIDA ---")
        print(f"Contexto Visual (BLIP): '{resultado['descripcion']}'")
        print(f"Speed LV detectado    : {resultado['speed_lv']}")
        print(f"Líneas detectadas     : {resultado['lines']}")
        print(f"Score detectado       : {resultado['score']}")
        print(f"Evento Especial       : {resultado['evento']}")
        print(f"Es Back-to-Back?      : {resultado['back_to_back']}")