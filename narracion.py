import os
import json
import random
import requests
import cv2
import torch
import numpy as np

from PIL import Image

from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor
)

from vision import ModuloVision


class narracion:

    def __init__(self):

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

        # =====================================================
        # MODULO VISION
        # =====================================================

        print("Cargando módulo de visión...")

        self.vision = ModuloVision()

        # =====================================================
        # MEMORIA NARRATIVA Y CONEXIÓN LLM
        # =====================================================

        print("Configurando conexión a Hugging Face API para el comentarista...")
        self.api_url = "https://api-inference.huggingface.co/models/Groq/Llama-3-Groq-8B-Tool-Use/v1/chat/completions"
        hf_token = os.environ.get("HF_TOKEN", "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")
        self.headers = {
            "Authorization": f"Bearer {hf_token}",
            "Content-Type": "application/json"
        }

        self.estado_narrativo = {
            "ultimo_evento": None,
            "racha_tetris": 0,
            "presion_consecutiva": 0,
            "frames_sin_evento": 0,
            "ultimo_peligro": "LOW"
        }

        self.historial = []
        self.combo = 0
        self.ultimo_comentario_frame = -999

        print("¡Narrador listo!")

    # =====================================================
    # VLM
    # =====================================================

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

    # =====================================================
    # GENERAR COMENTARIO (CON LLM Llama-3)
    # =====================================================

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
        print("Generando comentario dinámico con LLM...")

        # Construimos el contexto exacto de lo que está pasando
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

        # Historial para evitar que repita lo último que dijo
        historial_str = "\n".join(f"- {c}" for c in self.historial[-3:]) if self.historial else "None"

        system_prompt = """
        You are an incredibly energetic, professional hype-caster for a world championship Tetris match. 
        Your job is to read the game state and output an exciting, highly specific play-by-play commentary paragraph.
        
        STRICT RULES:
        1. Write AT LEAST 2 to 3 sentences. Make it long, emotional and descriptive.
        2. Mention specific numbers (like the score, lines, or speed level) to sound analytical.
        3. React to the 'Visual description' provided by the VLM.
        4. If a MAJOR EVENT (Tetris, T-Spin) or HIGH DANGER is happening, HYPE IT UP!
        5. DO NOT repeat phrases from the recent history.
        """

        user_prompt = f"CURRENT GAME STATE:\n{contexto_juego}\n\nRECENT HISTORY (Do not repeat these):\n{historial_str}\n\nGenerate your commentary now:"

        payload = {
            "model": "Groq/Llama-3-Groq-8B-Tool-Use",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 150,
            "temperature": 0.8
        }

        try:
            respuesta = requests.post(self.api_url, headers=self.headers, json=payload)
            
            if respuesta.status_code == 200:
                datos = respuesta.json()
                comentario = datos["choices"][0]["message"]["content"].strip()
                comentario = comentario.replace('"', "")

                self.historial.append(comentario)
                if len(self.historial) > 10:
                    self.historial.pop(0)

                return comentario
            else:
                print(f"Error API: {respuesta.status_code} - {respuesta.text}")
                return "The board is moving so fast! Let's see what happens next!"
        except Exception as e:
            print(f"Error en la petición: {e}")
            return "The pressure is immense, let's see how the player handles this!"

    # =====================================================
    # GENERAR COMENTARIOS PARA CARPETA (HEURÍSTICA ORIGINAL)
    # =====================================================

    def generar_comentarios_para_frames(
        self,
        carpeta_frames,
        intervalo_segundos=1.0,
        max_frames=None,
        output_json=None
    ):

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

        for idx, ruta in enumerate(rutas):

            print(f"Analizando frame {idx + 1}/{len(rutas)}")

            if idx % 5 != 0:
                continue

            estado = self._extraer_estado_frame(ruta)
            evento = estado.get("evento_vision")

            if evento is None:
                evento = self._detectar_evento(estado_previo, estado)

            if evento:
                self.estado_narrativo["ultimo_evento"] = evento
                self.estado_narrativo["frames_sin_evento"] = 0
            else:
                self.estado_narrativo["frames_sin_evento"] += 1

            if estado["danger"] == "HIGH":
                self.estado_narrativo["presion_consecutiva"] += 1
            else:
                self.estado_narrativo["presion_consecutiva"] = 0

            if evento == "TETRIS":
                self.estado_narrativo["racha_tetris"] += 1
            else:
                self.estado_narrativo["racha_tetris"] = 0

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

            if idx % 20 == 0:
                prioridad += 5

            if (
                estado_previo
                and estado_previo["danger"] == "HIGH"
                and estado["danger"] == "LOW"
            ):
                prioridad += 10
                evento = "RECOVERY"

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

            if estado_previo:
                delta_score = estado["score"] - estado_previo["score"]
                if delta_score > 3000:
                    prioridad += 6

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

    # =====================================================
    # EXTRAER ESTADO
    # =====================================================

    def _extraer_estado_frame(self, ruta_frame):

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

        estado["danger"] = self._calcular_peligro(imagen)

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

        descripcion_vlm = self.analizar_frame_vlm(ruta_frame)

        estado["descripcion"] = descripcion_vlm + " " + descripcion_blip

        return estado

    # =====================================================
    # EVENTOS Y PELIGRO
    # =====================================================

    def _detectar_evento(self, previo, actual):

        if previo is None:
            return None

        delta_lines = actual["lines"] - previo["lines"]
        delta_score = actual["score"] - previo["score"]

        if delta_lines >= 4: return "TETRIS"
        if delta_lines == 3: return "TRIPLE"
        if delta_lines == 2: return "DOUBLE"
        if delta_lines == 1: return "LINE CLEAR"
        if delta_score >= 1200: return "HOT STREAK"

        return None

    def _calcular_peligro(self, imagen):

        alto, ancho = imagen.shape[:2]

        tablero = imagen[
            int(alto * 0.18): int(alto * 0.82),
            int(ancho * 0.30): int(ancho * 0.55)
        ]

        gris = cv2.cvtColor(tablero, cv2.COLOR_BGR2GRAY)
        brillo = np.mean(gris)

        if brillo > 90: return "HIGH"
        if brillo > 60: return "MEDIUM"

        return "LOW"


# =========================================================
# MAIN
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