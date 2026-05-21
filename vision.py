from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import easyocr
import numpy as np
import re

class ModuloVision:
    def __init__(self):
        print("Cargando modelo BLIP localmente (Vía clases nativas)...")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        print("¡Modelo BLIP cargado!")

        print("Cargando lector OCR...")
        self.lector_ocr = easyocr.Reader(['en'], gpu=False) 
        print("¡Lector OCR listo para analizar!")

    def extraer_numeros(self, texto_leido):
        """
        Limpia el texto detectado para quedarse solo con los números.
        Ahora ignora comas y puntos para scores grandes (ej. '35,322' -> '35322').
        """
        if not texto_leido:
            return "0"
        
        # Unimos todo y borramos las comas o puntos que confunden al OCR
        texto_unido = " ".join(texto_leido).replace(",", "").replace(".", "")
        numeros = re.findall(r'\d+', texto_unido)
        return numeros[0] if numeros else "0"

    def clasificar_imagen(self, ruta_imagen):
        print(f"Analizando {ruta_imagen}...")
        
        try:
            imagen = Image.open(ruta_imagen).convert('RGB')
            
# --- PARTE A: LA "VIBRA" CON BLIP ---
            # Forzamos al modelo a entender que está viendo un juego
            texto_guia = "a game of Tetris showing"
            
            # Pasamos la imagen Y el texto guía al procesador
            inputs = self.processor(imagen, text=texto_guia, return_tensors="pt")
            
            out = self.model.generate(
                **inputs, 
                max_new_tokens=40,
                repetition_penalty=1.5 
            )
            descripcion = self.processor.decode(out[0], skip_special_tokens=True)
            
# --- PARTE B: LECTURA DE DATOS CON OCR ---
            
            # 1. Recorte del SPEED LV (Bajamos de 740->790 a 760->810)
            coordenadas_speed = (652, 790, 701, 832) 
            recorte_speed = imagen.crop(coordenadas_speed)
            recorte_speed.show()
            
            texto_speed_crudo = self.lector_ocr.readtext(np.array(recorte_speed), detail=0)
            speed_final = self.extraer_numeros(texto_speed_crudo)

            # 2. Recorte de las LINES (Bajamos de 840->890 a 860->910)
            coordenadas_lines = (642, 894, 702, 942) 
            recorte_lines = imagen.crop(coordenadas_lines)
            recorte_lines.show()
            
            texto_lines_crudo = self.lector_ocr.readtext(np.array(recorte_lines), detail=0)
            lines_final = self.extraer_numeros(texto_lines_crudo)

            # 3. Recorte del SCORE (Bajamos de 750->810 a 770->830)
            coordenadas_score = (1227, 789, 1425, 872) 
            recorte_score = imagen.crop(coordenadas_score)
            recorte_score.show()
            
            texto_score_crudo = self.lector_ocr.readtext(np.array(recorte_score), detail=0)
            score_final = self.extraer_numeros(texto_score_crudo)

            # 4. NUEVO: Recorte de EVENTOS ESPECIALES (Lado izquierdo, zona central)
            coordenadas_evento = (436, 312, 698, 570) 
            recorte_evento = imagen.crop(coordenadas_evento)
            recorte_evento.show() # Quita el '#' si necesitas ajustar esta nueva caja
            
            texto_evento_crudo = self.lector_ocr.readtext(np.array(recorte_evento), detail=0)
            # Unimos todo el texto y lo pasamos a mayúsculas para buscar coincidencias
            texto_evento_unido = " ".join(texto_evento_crudo).upper()
            
            evento_detectado = None
            if "TETR" in texto_evento_unido: # Buscamos solo "TETR" por si la 'I' o 'S' fallan
                evento_detectado = "TETRIS"
            elif "SPIN" in texto_evento_unido:
                evento_detectado = "T-SPIN"
                
            es_back_to_back = "BACK" in texto_evento_unido

            # 5. Devolvemos el paquete maestro
            return {
                "descripcion": descripcion,
                "speed_lv": int(speed_final),
                "lines": int(lines_final),
                "score": int(score_final),
                "evento": evento_detectado,
                "back_to_back": es_back_to_back
            }
                
        except FileNotFoundError:
            print(f"Error: No se encontró la imagen en la ruta '{ruta_imagen}'")
            return None
        except Exception as e:
            print(f"Error procesando la imagen: {e}")
            return None

# --- PRUEBA DEL MÓDULO ---
if __name__ == "__main__":
    vision = ModuloVision()
    
    # Cambia esto por el nombre exacto de la foto donde sale el T-Spin
    imagen_prueba = "frames_capturados/frame_0085.jpg" 
    
    resultado = vision.clasificar_imagen(imagen_prueba)
    
    if resultado:
        print("\n--- RESULTADO DE LA VISIÓN HÍBRIDA ---")
        print(f"Contexto Visual (BLIP): '{resultado['descripcion']}'")
        print(f"Speed LV detectado    : {resultado['speed_lv']}")
        print(f"Líneas detectadas     : {resultado['lines']}")
        print(f"Score detectado       : {resultado['score']}")
        print(f"Evento Especial       : {resultado['evento']}")
        print(f"Es Back-to-Back?      : {resultado['back_to_back']}")