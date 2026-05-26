# ====== IMPORTS ======
import re
import torch
import scipy.io.wavfile
from transformers import VitsModel, AutoTokenizer

class ModuloAudio:

    # ====== INICIALIZAR MODELO DE AUDIO ======
    # Detecta entorno y carga el modelo VITS para Text-to-Speech
    def __init__(self):
        print("Configurando módulo de Text-to-Speech (Local VITS - MMS Inglés ultrarrápido)...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        modelo = "facebook/mms-tts-eng"
        
        self.tokenizer = AutoTokenizer.from_pretrained(modelo)
        self.model = VitsModel.from_pretrained(modelo).to(self.device)
        self.sample_rate = self.model.config.sampling_rate
        
        print(f"¡Modelo {modelo} cargado en memoria ({self.device})!")

    # ====== GENERAR AUDIO (TTS) ======
    # Pre-procesa el texto y genera un archivo de audio local usando VITS
    def generar_audio(self, texto, ruta_salida="comentario_temporal.wav"):
        try:
            texto_limpio = texto.lower()
            texto_limpio = texto_limpio.replace("-", " ")
            texto_limpio = re.sub(r'[^a-z \.,\?\!\']', '', texto_limpio)
            texto_limpio = re.sub(r'\s+', ' ', texto_limpio).strip()
            
            print(f"Generando voz local rápida para: '{texto_limpio[:40]}...'")
            
            inputs = self.tokenizer(texto_limpio, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                salida = self.model(**inputs).waveform
            
            audio_numpy = salida[0].cpu().numpy()
            scipy.io.wavfile.write(ruta_salida, self.sample_rate, audio_numpy)
                
            return ruta_salida
            
        except Exception as e:
            print(f"Error generando el audio TTS local: {repr(e)}")
            return None

# =========================================================
# PRUEBA DEL MÓDULO (MAIN)
# =========================================================
if __name__ == "__main__":
    audio = ModuloAudio()
    texto_prueba = "Tetris again! And with back to back, it is over sixty thousand now!"
    ruta = audio.generar_audio(texto_prueba)
    if ruta:
        print(f"¡Éxito! El audio veloz se guardó correctamente en: {ruta}")