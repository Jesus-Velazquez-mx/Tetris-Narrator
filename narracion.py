# ====== IMPORTS ======
import os
import json
import random
import cv2
import torch
import numpy as np

from PIL import Image

from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor
)

from huggingface_hub import InferenceClient

from vision import ModuloVision


class narracion:

    def __init__(self):
        # ====== CARGAR MODELO VLM (SmolVLM) ======
        # Detecta GPU y carga modelo de visión-lenguaje local
        print("Cargando SmolVLM...")

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        modelo = (
            "HuggingFaceTB/SmolVLM-500M-Instruct"
        )

        self.processor = (
            AutoProcessor.from_pretrained(
                modelo
            )
        )

        self.vlm = (
            AutoModelForImageTextToText
            .from_pretrained(
                modelo,
                torch_dtype=(
                    torch.float16
                    if self.device == "cuda"
                    else torch.float32
                )
            )
            .to(self.device)
        )

        self.vlm.eval()

        # ====== CARGAR MÓDULO DE VISIÓN ======
        # Usa OCR y BLIP para extraer score, nivel y eventos
        print("Cargando módulo de visión...")

        self.vision = ModuloVision()

        # ====== CONEXIÓN A LLM (HUGGING FACE) ======
        # Cliente para generar comentarios dinámicos en tiempo real
        print("Configurando cliente oficial de Hugging Face SDK para el comentarista...")
        
        hf_token = os.environ.get("HF_TOKEN", "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")
        self.cliente_hf = InferenceClient(token=hf_token)

        # ====== INICIALIZAR ESTADO Y MEMORIA ======
        # Rastrear eventos, combos y evitar repetición en comentarios
        self.estado_narrativo = {
            "ultimo_evento": None,
            "racha_tetris": 0,
            "presion_consecutiva": 0,
            "frames_sin_evento": 0,
            "ultimo_peligro": "LOW"
        }

        self.historial = []  # Últimos comentarios para evitar repetición
        self.combo = 0  # Contador de eventos consecutivos
        self.ultimo_comentario_frame = -999  # Cooldown entre comentarios

        print("¡Narrador listo!")

    # ====== ANÁLISIS CON MODELO VLM ======
    # Describe el frame en una oración usando SmolVLM
    def analizar_frame_vlm(self, ruta_imagen):

        try:

            imagen = (
                Image
                .open(ruta_imagen)
                .convert("RGB")
                .resize((384, 384))
            )

            prompt = """
Analyze this competitive Tetris frame.

Focus ONLY on:
- dangerous stacks
- pressure
- clean stacking
- recoveries
- near top out
- aggressive play

Respond in ONE short sentence.
"""

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image"
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]

            texto_prompt = (
                self.processor
                .apply_chat_template(
                    messages,
                    add_generation_prompt=True
                )
            )

            inputs = self.processor(
                text=texto_prompt,
                images=imagen,
                return_tensors="pt"
            )

            inputs = {
                k: v.to(self.device)
                for k, v in inputs.items()
            }

            # Generar texto del modelo
            with torch.no_grad():

                generated_ids = (
                    self.vlm.generate(
                        **inputs,
                        max_new_tokens=20,
                        do_sample=False
                    )
                )

            salida = (
                self.processor
                .batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )[0]
            )

            return salida.lower()

        except Exception as e:

            print(f"Error VLM: {e}")

            return ""

    # ====== GENERAR COMENTARIO CON LLM ======
    # Usa eventos y contexto para generar narración dinámica y única
    def generar_comentario(
        self,
        descripcion_vision,
        speed_lv,
        lines,
        score,
        evento=None,
        danger="LOW",
        back_to_back=False
    ):
        import random 
        
        print("Generando comentario dinámico con Hugging Face SDK...")

        # Construir contexto con estado actual del juego
        contexto_juego = f"""
        - Current Speed Level: {speed_lv}
        - Lines Cleared: {lines}
        - Current Score: {score}
        - Board Danger Level: {danger}
        - Visual description: {descripcion_vision}
        """
        
        if evento:
            contexto_juego += f"\n- MAJOR EVENT JUST HAPPENED: {evento}!"
        if back_to_back:
            contexto_juego += f"\n- BACK-TO-BACK BONUS IS ACTIVE!"

        historial_str = "\n".join(f"- {c}" for c in self.historial[-3:]) if self.historial else "None"

        # Priorizar eventos sobre relleno: variar ángulo de comentario
        if evento or back_to_back:
            elementos_evento = []
            if evento: elementos_evento.append(str(evento).replace("T-SPIN", "Tee Spin"))
            if back_to_back: elementos_evento.append("Back to Back")
            
            nombres_eventos = " and ".join(elementos_evento)
            
            # Múltiples perspectivas del mismo evento
            enfoques_evento = [
                f"the raw hype and adrenaline of that {nombres_eventos}",
                f"how that {nombres_eventos} completely shifts the game's momentum",
                f"the strategic brilliance of executing that {nombres_eventos}",
                f"a calm, classy acknowledgment of the {nombres_eventos}"
            ]
            tema_elegido = random.choice(enfoques_evento)
        else:
            # Comentarios generales sin evento
            topicos = [
                "the geometry and shape of the block structure",
                "the tension and survival strategy",
                "the visual colors of the board",
                "the pacing and momentum of the drops",
                "the anticipation of what piece comes next"
            ]
            tema_elegido = random.choice(topicos)

        system_prompt = """
        You are a stylish and observant commentator for a Tetris match. You speak with a calm, elegant confidence using SIMPLE, EVERYDAY WORDS.
        
        STRICT RULES:
        1. FORMAT: Write exactly 1 short sentence. Maximum 25 words.
        2. THE NO-NUMBER RULE (CRITICAL): YOU MUST NEVER USE NUMBERS. Do not read scores, lines, or speed levels.
        3. MANDATORY TERMINOLOGY: If the game state mentions an event, YOU MUST INCLUDE the exact words (like "Tetris", "Tee Spin", or "Back to Back"). CRITICAL: Integrate them fluidly into your unique sentence.
        4. PLAYER IDENTITY: Refer to the human as "the player" or focus directly on the board.
        5. AVOID REPETITION: Check the RECENT HISTORY. Your new sentence MUST NOT use the same verbs, adjectives, or structure as the previous comments.
        """

        user_prompt = f"CURRENT GAME STATE:\n{contexto_juego}\n\nRECENT HISTORY:\n{historial_str}\n\nCRITICAL INSTRUCTION: Focus your comment EXCLUSIVELY on {tema_elegido}.\n\nGenerate your NEW, UNIQUE short commentary now:"

        mensajes = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            # Llamar a LLM con prompts optimizados para Tetris
            respuesta = self.cliente_hf.chat_completion(
                model="Qwen/Qwen2.5-7B-Instruct",
                messages=mensajes,
                max_tokens=22,
                temperature=0.7,      # Control de creatividad
                frequency_penalty=1.5  # Penalizar repetición
            )
            
            comentario = respuesta.choices[0].message.content.strip().replace('"', "")

            # Guardar en historial para evitar repetición
            self.historial.append(comentario)
            if len(self.historial) > 10:
                self.historial.pop(0)

            return comentario
            
        except Exception as e:
            print(f"Error en Hugging Face SDK: {e}")
            return "The game continues with intense pressure."

    # ====== PROCESAR TODOS LOS FRAMES ======
    # Loop principal: analiza frames, detecta eventos y genera comentarios
    def generar_comentarios_para_frames(
        self,
        carpeta_frames,
        intervalo_segundos=1.0,
        max_frames=None,
        output_json=None
    ):

        # Cargar todas las imágenes de la carpeta
        rutas = sorted([
            os.path.join(carpeta_frames, f)
            for f in os.listdir(carpeta_frames)
            if f.lower().endswith(
                (
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".bmp"
                )
            )
        ])

        if max_frames:
            rutas = rutas[:max_frames]

        comentarios = []
        estado_previo = None

        # Procesar cada frame
        for idx, ruta in enumerate(rutas):

            print(f"Analizando frame {idx + 1}/{len(rutas)}")

            # Procesar cada 5 frames para reducir carga
            if idx % 5 != 0:
                continue

            # Extraer datos: OCR, visión y descripción
            estado = self._extraer_estado_frame(ruta)
            evento = estado.get("evento_vision")

            # Detectar evento si la visión no lo hizo
            if evento is None:
                evento = self._detectar_evento(estado_previo, estado)

            # Actualizar estado narrativo
            if evento:
                self.estado_narrativo["ultimo_evento"] = evento
                self.estado_narrativo["frames_sin_evento"] = 0
            else:
                self.estado_narrativo["frames_sin_evento"] += 1

            # Rastrear presión (cuántos frames consecutivos en HIGH)
            if estado["danger"] == "HIGH":
                self.estado_narrativo["presion_consecutiva"] += 1
            else:
                self.estado_narrativo["presion_consecutiva"] = 0

            # Rastrear Tetris consecutivos
            if evento == "TETRIS":
                self.estado_narrativo["racha_tetris"] += 1
            else:
                self.estado_narrativo["racha_tetris"] = 0

            # ====== CÁLCULO DE PRIORIDAD ======
            # Determina si vale la pena generar comentario
            prioridad = 0

            if evento == "TETRIS":
                prioridad += 10
            elif evento == "T-SPIN":
                prioridad += 9
            elif evento == "TRIPLE":
                prioridad += 6
            elif evento == "DOUBLE":
                prioridad += 4

            if estado["danger"] == "HIGH":
                prioridad += 8

            if estado.get("back_to_back"):
                prioridad += 5

            if idx % 20 == 0:  # Comentario cada ~20 frames
                prioridad += 5

            # Recuperación de peligro alto
            if (
                estado_previo
                and estado_previo["danger"] == "HIGH"
                and estado["danger"] == "LOW"
            ):
                prioridad += 10
                evento = "RECOVERY"

            # Rastrear combo (eventos consecutivos)
            if evento in {
                "LINE CLEAR",
                "DOUBLE",
                "TRIPLE",
                "TETRIS",
                "T-SPIN"
            }:
                self.combo += 1
            else:
                self.combo = 0

            if self.combo >= 3:
                prioridad += 5

            # Detectar "hot streaks" (grandes aumentos de score)
            if estado_previo:
                delta_score = estado["score"] - estado_previo["score"]
                if delta_score > 3000:
                    prioridad += 6

            # Cooldown: no comentar demasiado seguido
            cooldown_frames = 8

            if (idx - self.ultimo_comentario_frame < cooldown_frames):
                prioridad = 0

            debe_comentar = (prioridad >= 5)

            if not debe_comentar:
                estado_previo = estado
                self.estado_narrativo["ultimo_peligro"] = estado["danger"]
                continue

            comentario = self.generar_comentario(
                descripcion_vision=estado["descripcion"],
                speed_lv=estado["level"],
                lines=estado["lines"],
                score=estado["score"],
                evento=evento,
                danger=estado["danger"],
                back_to_back=estado.get("back_to_back", False)
            )

            self.ultimo_comentario_frame = idx

            registro = {
                "frame": os.path.basename(ruta),
                "time_sec": round(idx * intervalo_segundos, 2),
                "level": estado["level"],
                "lines": estado["lines"],
                "score": estado["score"],
                "danger": estado["danger"],
                "evento": evento,
                "descripcion_vlm": estado["descripcion"],
                "commentary": comentario
            }

            comentarios.append(registro)
            estado_previo = estado
            self.estado_narrativo["ultimo_peligro"] = estado["danger"]

        if output_json:
            with open(output_json, "w", encoding="utf-8") as archivo:
                json.dump(comentarios, archivo, indent=2, ensure_ascii=False)
            print(f"Comentarios guardados en {output_json}")

        return comentarios

    # ====== EXTRAER ESTADO DEL FRAME ======
    # Obtiene score, nivel, líneas, peligro y descripción visual
    def _extraer_estado_frame(self, ruta_frame):
        # Estado por defecto
        estado = {
            "level": 1,
            "lines": 0,
            "score": 0,
            "danger": "LOW",
            "descripcion": "",
            "evento_vision": None,
            "back_to_back": False
        }

        imagen = cv2.imread(ruta_frame)
        if imagen is None:
            return estado

        # Calcular peligro visual (brillo del tablero)
        estado["danger"] = self._calcular_peligro(imagen)

        # Extraer OCR: score, nivel, líneas
        resultado_vision = self.vision.clasificar_imagen(ruta_frame)
        if resultado_vision:
            estado["level"] = resultado_vision["speed_lv"]
            estado["lines"] = resultado_vision["lines"]
            estado["score"] = resultado_vision["score"]
            estado["evento_vision"] = resultado_vision["evento"]
            estado["back_to_back"] = resultado_vision["back_to_back"]
            descripcion_blip = resultado_vision["descripcion"]
        else:
            descripcion_blip = ""

        # Descripción visual con VLM
        descripcion_vlm = self.analizar_frame_vlm(ruta_frame)
        estado["descripcion"] = descripcion_vlm + " " + descripcion_blip

        return estado

    # ====== DETECTAR EVENTOS (Cambios de estado) ======
    # Compara frames anteriores y actuales
    def _detectar_evento(self, previo, actual):
        if previo is None:
            return None

        # Comparar líneas despejadas
        delta_lines = actual["lines"] - previo["lines"]
        delta_score = actual["score"] - previo["score"]

        if delta_lines >= 4: return "TETRIS"
        if delta_lines == 3: return "TRIPLE"
        if delta_lines == 2: return "DOUBLE"
        if delta_lines == 1: return "LINE CLEAR"
        if delta_score >= 1200: return "HOT STREAK"

        return None

    # ====== CALCULAR NIVEL DE PELIGRO ======
    # Análisis visual del brillo del tablero
    def _calcular_peligro(self, imagen):
        alto, ancho = imagen.shape[:2]

        # Extraer región del tablero
        tablero = imagen[
            int(alto * 0.18): int(alto * 0.82),
            int(ancho * 0.30): int(ancho * 0.55)
        ]

        # Calcular brillo promedio (más blanco = más peligroso)
        gris = cv2.cvtColor(tablero, cv2.COLOR_BGR2GRAY)
        brillo = np.mean(gris)

        if brillo > 90: return "HIGH"
        if brillo > 60: return "MEDIUM"
        return "LOW"


# =========================================================
# PRUEBA DEL MÓDULO (MAIN)
# =========================================================

def generar_comentarios_desde_carpeta(
    carpeta_frames,
    output_json="comentarios_frames.json"
):
    narrador = narracion()
    return narrador.generar_comentarios_para_frames(
        carpeta_frames,
        output_json=output_json
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Narrador IA de Tetris.")
    parser.add_argument("carpeta_frames", help="Ruta de frames")
    parser.add_argument("--intervalo", type=float, default=1.0)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--output", default="comentarios_frames.json")

    args = parser.parse_args()
    narrador = narracion()

    comentarios = narrador.generar_comentarios_para_frames(
        args.carpeta_frames,
        intervalo_segundos=args.intervalo,
        max_frames=args.max_frames,
        output_json=args.output
    )

    print(f"Se generaron {len(comentarios)} comentarios.")

    for registro in comentarios[:10]:
        print(f"[{registro['frame']}] {registro['commentary']}")