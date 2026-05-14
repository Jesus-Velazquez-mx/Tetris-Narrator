from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

class ModuloVision:
    def __init__(self):
        """
        Inicializa el módulo descargando el modelo BLIP de forma nativa.
        """
        print("Cargando modelo BLIP localmente (Vía clases nativas)...")
        
        # En lugar del pipeline genérico, instanciamos el procesador y el modelo de BLIP directamente
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        print("¡Modelo BLIP cargado y listo para analizar!")

    def clasificar_imagen(self, ruta_imagen):
        """
        Analiza la imagen y genera una descripción en texto.
        """
        print(f"Analizando {ruta_imagen}...")
        
        try:
            # 1. Abrimos la imagen y aseguramos que esté en formato RGB
            imagen = Image.open(ruta_imagen).convert('RGB')
            
            # 2. Pre-procesamos la imagen para que el modelo la entienda
            inputs = self.processor(imagen, return_tensors="pt")
            
            # 3. Le pedimos al modelo que genere texto
            out = self.model.generate(
                **inputs, 
                max_new_tokens=40,         # Le damos permiso de escribir hasta 40 palabras
                repetition_penalty=1.5     # Si repite sílabas (como "tet tet"), la IA lo cancela
            )
            
            # 4. Decodificamos la respuesta
            descripcion = self.processor.decode(out[0], skip_special_tokens=True)
            
            return descripcion
                
        except FileNotFoundError:
            print(f"Error: No se encontró la imagen en la ruta '{ruta_imagen}'")
            return None
        except Exception as e:
            print(f"Error procesando la imagen: {e}")
            return None

# --- PRUEBA DEL MÓDULO ---
if __name__ == "__main__":
    vision = ModuloVision()
    
    imagen_prueba = "frames_capturados/frame_0100.jpg" 
    
    resultado = vision.clasificar_imagen(imagen_prueba)
    
    if resultado:
        print("\n--- RESULTADO DE LA VISIÓN ---")
        print(f"La IA describe la imagen como: '{resultado}'")