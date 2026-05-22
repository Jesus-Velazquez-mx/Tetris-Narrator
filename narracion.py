import os
import re
import json
import random

import cv2
import numpy as np

try:
    import pytesseract

    # IMPORTANTE:
    # Ajusta esta ruta si Tesseract está en otra ubicación
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )

except ImportError:
    pytesseract = None

try:
    from groq import Groq
except ImportError:
    Groq = None


class narracion:

    def __init__(
        self,
        llm_provider="groq",
        model="llama-3.1-8b-instant",
        ocr_enabled=True
    ):

        self.ocr_enabled = ocr_enabled and pytesseract is not None

        self.historial = []

        self.model = model
        self.llm_provider = llm_provider

        self.client = None

        if self.llm_provider == "groq" and Groq is not None:

            print("Inicializando narrador con Groq...")

            self.client = Groq(
                api_key=os.environ.get("GROQ_API_KEY")
            )

        else:
            print(
                "Groq no disponible o no configurado. "
                "Se usará fallback local."
            )

        # -----------------------------------------
        # OCR
        # -----------------------------------------

        if self.ocr_enabled:

            if self._tesseract_instalado():

                print(
                    "OCR activado correctamente."
                )

            else:

                self.ocr_enabled = False

                print(
                    "Tesseract no encontrado. OCR desactivado."
                )

        else:

            print(
                "OCR desactivado."
            )

        print("¡Módulo de Narración listo!")

    # =====================================================
    # TESSERACT
    # =====================================================

    def _tesseract_instalado(self):

        if pytesseract is None:
            return False

        try:
            pytesseract.get_tesseract_version()
            return True

        except Exception:
            return False

    # =====================================================
    # COMENTARIO IA
    # =====================================================

    def generar_comentario(
        self,
        descripcion_vision,
        speed_lv,
        lines,
        score,
        evento=None,
        back_to_back=False,
        danger="LOW"
    ):

        if self.client is None:

            return self._fallback(
                evento,
                speed_lv,
                lines,
                score,
                descripcion_vision
            )

        print("Generando comentario con LLM...")

        partes_contexto = [
            f"Level: {speed_lv}",
            f"Score: {score}",
            f"Lines cleared: {lines}",
            f"Danger level: {danger}",
        ]

        if evento:
            partes_contexto.append(
                f"Event: {evento}"
            )

        if back_to_back:
            partes_contexto.append(
                "Back-to-back active"
            )

        contexto_juego = "\n".join(partes_contexto)

        historial_str = ""

        if self.historial:

            recientes = "\n".join(
                f"- {c}" for c in self.historial[-4:]
            )

            historial_str = (
                "\nDo NOT repeat these comments:\n"
                f"{recientes}"
            )

        # -----------------------------------------
        # PROMPT MEJORADO
        # -----------------------------------------

        system_prompt = """
You are a professional Tetris esports commentator.

STRICT RULES:
- Speak ONLY about Tetris.
- Maximum 1 short sentence.
- Sound energetic and realistic.
- Do not invent fake events.
- Do not repeat previous commentary.
- Focus on:
  - speed
  - danger
  - survival
  - clean stacking
  - pressure
  - tetris clears
  - combos

GOOD examples:
- What a clean Tetris under pressure!
- The stack is getting dangerously high!
- Incredible speed at this level!
- That recovery was perfect!

BAD examples:
- The battlefield explodes
- The crowd uses magic
- The player enters another dimension

Return ONLY the commentary line.
"""

        user_prompt = (
            f"Current game state:\n"
            f"{contexto_juego}"
            f"{historial_str}\n\n"
            "Generate commentary."
        )

        try:

            respuesta = self.client.chat.completions.create(

                model=self.model,

                max_tokens=40,

                temperature=0.7,

                top_p=0.9,

                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            )

            comentario = (
                respuesta
                .choices[0]
                .message
                .content
                .strip()
            )

            comentario = comentario.replace('"', "")

            self.historial.append(comentario)

            if len(self.historial) > 12:
                self.historial.pop(0)

            return comentario

        except Exception as e:

            print(f"Error generando narración: {e}")

            return self._fallback(
                evento,
                speed_lv,
                lines,
                score,
                descripcion_vision
            )

    # =====================================================
    # GENERAR COMENTARIOS
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
                (".png", ".jpg", ".jpeg", ".bmp")
            )
        ])

        if max_frames:
            rutas = rutas[:max_frames]

        if not rutas:

            print(
                f"No se encontraron frames en {carpeta_frames}"
            )

            return []

        comentarios = []

        estado_previo = None
        evento_previo = None

        for idx, ruta in enumerate(rutas):

            estado = self._extraer_estado_frame(ruta)

            evento = self._detectar_evento(
                estado_previo,
                estado
            )

            back_to_back = self._es_back_to_back(
                evento_previo,
                evento
            )

            # -----------------------------------------
            # SOLO comentar momentos importantes
            # -----------------------------------------

            debe_comentar = (
                evento is not None
                or estado["danger"] == "HIGH"
                or idx % 8 == 0
            )

            if not debe_comentar:

                estado_previo = estado
                evento_previo = evento

                continue

            descripcion = (
                f"Frame {idx + 1} del gameplay."
            )

            comentario = self.generar_comentario(
                descripcion_vision=descripcion,
                speed_lv=estado.get("level", 1),
                lines=estado.get("lines", 0),
                score=estado.get("score", 0),
                evento=evento,
                back_to_back=back_to_back,
                danger=estado.get("danger", "LOW")
            )

            registro = {

                "frame": os.path.basename(ruta),

                "time_sec": round(
                    idx * intervalo_segundos,
                    2
                ),

                "level": estado.get("level", 1),

                "lines": estado.get("lines", 0),

                "score": estado.get("score", 0),

                "danger": estado.get("danger"),

                "evento": evento,

                "commentary": comentario,
            }

            comentarios.append(registro)

            estado_previo = estado
            evento_previo = evento

        # -----------------------------------------
        # GUARDAR JSON
        # -----------------------------------------

        if output_json:

            with open(
                output_json,
                "w",
                encoding="utf-8"
            ) as archivo:

                json.dump(
                    comentarios,
                    archivo,
                    indent=2,
                    ensure_ascii=False
                )

            print(
                f"Comentarios guardados en {output_json}"
            )

        return comentarios

    # =====================================================
    # EXTRAER ESTADO
    # =====================================================

    def _extraer_estado_frame(self, ruta_frame):

        estado = {
            "level": 1,
            "lines": 0,
            "score": 0,
            "danger": "LOW"
        }

        if not os.path.exists(ruta_frame):
            return estado

        imagen = cv2.imread(ruta_frame)

        if imagen is None:
            return estado

        # -----------------------------------------
        # PELIGRO VISUAL
        # -----------------------------------------

        estado["danger"] = (
            self._calcular_peligro(imagen)
        )

        # -----------------------------------------
        # OCR
        # -----------------------------------------

        if self.ocr_enabled:

            try:

                gris = cv2.cvtColor(
                    imagen,
                    cv2.COLOR_BGR2GRAY
                )

                gris = cv2.resize(
                    gris,
                    None,
                    fx=2,
                    fy=2
                )

                _, umbral = cv2.threshold(
                    gris,
                    150,
                    255,
                    cv2.THRESH_BINARY
                )

                texto = pytesseract.image_to_string(
                    umbral,
                    lang="eng"
                )

                texto = texto.replace("|", "1")

                datos = self._parsear_texto_ocr(
                    texto
                )

                estado.update(datos)

            except Exception as e:

                print(
                    f"OCR falló para {ruta_frame}: {e}"
                )

        return estado

    # =====================================================
    # OCR
    # =====================================================

    def _parsear_texto_ocr(self, texto):

        estado = {}

        texto = texto.lower()

        score = self._buscar_numero(
            texto,
            ["score", "points"]
        )

        lines = self._buscar_numero(
            texto,
            ["lines"]
        )

        level = self._buscar_numero(
            texto,
            ["level", "lv"]
        )

        if score is not None:
            estado["score"] = score

        if lines is not None:
            estado["lines"] = lines

        if level is not None:
            estado["level"] = level

        return estado

    def _buscar_numero(self, texto, claves):

        for clave in claves:

            if clave in texto:

                regex = re.compile(
                    rf"{re.escape(clave)}\D*([0-9][0-9\.,]*)"
                )

                m = regex.search(texto)

                if m:

                    return int(
                        m.group(1)
                        .replace(".", "")
                        .replace(",", "")
                    )

        return None

    # =====================================================
    # DETECCIÓN DE EVENTOS
    # =====================================================

    def _detectar_evento(self, previo, actual):

        if previo is None:
            return None

        delta_lines = (
            actual.get("lines", 0)
            - previo.get("lines", 0)
        )

        delta_score = (
            actual.get("score", 0)
            - previo.get("score", 0)
        )

        if delta_lines >= 4:
            return "TETRIS"

        if delta_lines == 3:
            return "TRIPLE"

        if delta_lines == 2:
            return "DOUBLE"

        if delta_lines == 1:
            return "LINE CLEAR"

        if delta_score >= 1200:
            return "HOT STREAK"

        return None

    def _es_back_to_back(self, anterior, actual):

        return (
            anterior in {"TETRIS", "T-SPIN"}
            and actual in {"TETRIS", "T-SPIN"}
        )

    # =====================================================
    # DETECCIÓN DE PELIGRO
    # =====================================================

    def _calcular_peligro(self, imagen):

        gris = cv2.cvtColor(
            imagen,
            cv2.COLOR_BGR2GRAY
        )

        pixeles_oscuros = np.sum(gris < 70)

        total = gris.size

        ratio = pixeles_oscuros / total

        if ratio > 0.55:
            return "HIGH"

        if ratio > 0.35:
            return "MEDIUM"

        return "LOW"

    # =====================================================
    # FALLBACK
    # =====================================================

    def _fallback(
        self,
        evento=None,
        speed_lv=1,
        lines=0,
        score=0,
        descripcion_vision=""
    ):

        if evento == "TETRIS":
            return (
                f"Massive Tetris at level {speed_lv}!"
            )

        if evento == "TRIPLE":
            return (
                "Strong triple line clear!"
            )

        if evento == "DOUBLE":
            return (
                "Clean double under pressure!"
            )

        frases = [

            f"The pace at level {speed_lv} is intense!",

            f"{lines} lines cleared so far.",

            f"The board is getting dangerous!",

            f"Current score: {score}.",

        ]

        return random.choice(frases)


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

    parser = argparse.ArgumentParser(
        description="Narrador IA de Tetris."
    )

    parser.add_argument(
        "carpeta_frames",
        help="Ruta de frames"
    )

    parser.add_argument(
        "--intervalo",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--max_frames",
        type=int,
        default=None
    )

    parser.add_argument(
        "--output",
        default="comentarios_frames.json"
    )

    args = parser.parse_args()

    narrador = narracion()

    comentarios = (
        narrador.generar_comentarios_para_frames(
            args.carpeta_frames,
            intervalo_segundos=args.intervalo,
            max_frames=args.max_frames,
            output_json=args.output,
        )
    )

    print(
        f"Se generaron {len(comentarios)} comentarios."
    )

    for registro in comentarios[:10]:

        print(
            f"[{registro['frame']}] "
            f"{registro['commentary']}"
        )