from transformers import pipeline
import random
import re

class ModuloNarracion:
    def __init__(self):
        print("Cargando modelo de NLP (GPT-2) localmente...")
        self.nlp = pipeline("text-generation", model="gpt2")
        print("¡Modelo de Narración listo!")

    def generar_comentario(self, descripcion_vision, speed_lv, lines, score, evento=None, back_to_back=False):
        print("Generando comentario (Modo Cebo + Contexto Estricto)...")
        
        # 1. Armamos el contexto de la jugada
        contexto = f"Level {speed_lv}, {score} points."
        if evento:
            contexto += f" A {evento}!"
        if back_to_back:
            contexto += " Back-to-Back!"
            
        # 2. EL CEBO
        inicios_posibles = [
            "This is absolutely",
            "I cannot believe",
            "What a brilliant",
            "Look at that"
        ]
        inicio_elegido = random.choice(inicios_posibles)
        
        # 3. EL SUPER-PROMPT: Le damos el formato exacto que debe copiar.
        prompt = (
            "Live transcript of a professional Tetris esports commentator.\n\n"
            "Game: Level 5, 12000 points. A TETRIS!\n"
            "Commentator: \"This is absolutely fantastic! A clean four-line clear!\"\n\n"
            "Game: Level 9, 35000 points. A T-SPIN!\n"
            "Commentator: \"Look at that piece rotation! Pure genius on the board!\"\n\n"
            "Game: Level 10, 50000 points.\n"
            "Commentator: \"I cannot believe how fast the blocks are falling right now!\"\n\n"
            f"Game: {contexto}\n"
            f"Commentator: \"{inicio_elegido}"
        )
        
        try:
            resultados = self.nlp(
                prompt,
                max_new_tokens=15,       # Reducido a 15 para evitar que divague o repita
                temperature=0.6,         # Más bajo = menos "alucinaciones" de cosas que no son Tetris
                repetition_penalty=1.3,  # Castigo estricto para que no repita "INSANE!"
                pad_token_id=50256,
                return_full_text=False   
            )
            
            # Agarramos el texto en crudo
            texto_bruto = resultados[0]['generated_text']
            
            # LIMPIEZA: Cortamos en el primer salto de línea o en la comilla de cierre
            texto_limpio = re.split(r'[\n"]', texto_bruto)[0].strip()
            
            if len(texto_limpio) < 2:
                texto_limpio = "amazing placement of the blocks"
                
            # Ensamblamos y gritamos
            comentario_final = f"{inicio_elegido} {texto_limpio}!!!".upper()
            
            return comentario_final
            
        except Exception as e:
            print(f"Error generando la narración: {e}")
            return "WHAT AN INTENSE MOMENT IN THE TETRIS MATCH!!!"

# --- PRUEBA DEL MÓDULO ---
if __name__ == "__main__":
    narrador = ModuloNarracion()
    
    descripcion = "a game of tetris"
    velocidad = 11
    lineas_totales = 30
    puntaje = 109094
    evento_ocr = "T-SPIN"
    es_btb = True
    
    print("\nGenerando 3 comentarios distintos con GPT-2 (Cebo + Few-Shot):")
    for i in range(3):
        comentario = narrador.generar_comentario(
            descripcion_vision=descripcion, 
            speed_lv=velocidad,
            lines=lineas_totales,
            score=puntaje, 
            evento=evento_ocr, 
            back_to_back=es_btb
        )
        print(f"\nToma {i+1}:")
        print(f"🎙️ {comentario}")