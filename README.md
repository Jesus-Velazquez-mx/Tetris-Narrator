# Narrador IA de Tetris (Esports Caster Bot)

Este proyecto es un sistema de Inteligencia Artificial capaz de analizar partidas de Tetris en video y generar comentarios de voz dinámicos en tiempo real. Orquesta múltiples ramas de la IA (Visión por Computadora, Procesamiento de Lenguaje Natural y Generación de Audio) dentro de una interfaz web funcional.

## Arquitectura del Sistema y Modelos Elegidos

El sistema procesa el video fragmentándolo en *frames* a 1 FPS y utiliza un pipeline de modelos especializados para "ver", "pensar" y "hablar". Se superó el requerimiento base al implementar una arquitectura de **3 modelos nativos de Hugging Face**.

### 1. Módulo de Captura y Procesamiento Visual
*   **Captura:** `OpenCV` extrae cuadros del archivo `.mp4` proporcionado por el usuario.
*   **OCR:** `EasyOCR` lee dinámicamente el puntaje (Score), las líneas limpiadas (Lines), el nivel de velocidad (Speed LV) y los eventos especiales (TETRIS, T-SPIN, BACK-TO-BACK) que aparecen en la interfaz.

### 2. Módulo de Visión (Híbrido)
Se optó por una solución híbrida multimodelo (Opción B de la rúbrica) para obtener un contexto absoluto del tablero:
*   **Modelo de Captioning:** `Salesforce/blip-image-captioning-base` (Hugging Face) para obtener la descripción base de la imagen.
*   **Modelo VLM:** `HuggingFaceTB/SmolVLM-500M-Instruct` (Hugging Face) configurado localmente para evaluar la presión, limpieza de estructura y peligro inminente (*near top out*).
*   **Algoritmo Clásico:** Análisis de matrices de píxeles y brillo mediante `NumPy` y `cv2` para determinar el nivel de "peligro" matemático de la torre de bloques.

### 3. Módulo de Narración (NLP)
*   **Modelo NLP:** `Qwen/Qwen2.5-7B-Instruct` (consumido vía Hugging Face SDK).
*   **Lógica:** Toma el contexto visual, el OCR y el historial de jugadas. Se aplican técnicas de *Prompt Engineering* avanzado con inyección dinámica de tópicos y reglas anti-números estrictas para obligar al LLM a comportarse como un comentarista deportivo, narrando *momentum* y estrategia sin repetir frases.

### 4. Módulo de Audio (Text-to-Speech)
*   **Modelo TTS:** `facebook/mms-tts-eng` (Hugging Face).
*   **Lógica:** Implementado de forma local mediante `VitsModel`. Se eligió este modelo por su capacidad de generar voces veloces y enérgicas, ideales para el ritmo frenético de los *esports*.

## Heurística y Lógica de Control (Caster Hype System)

El bot no narra cada *frame* a ciegas. Implementa un motor de toma de decisiones avanzado para determinar **cuándo comentar** y **cuándo callar**:

*   **Buzón de Eventos y Agrupación:** Si ocurren múltiples jugadas rápidas (ej. un *T-Spin* seguido de un *Back-to-Back*), el sistema retiene los eventos en memoria temporal y los combina inteligentemente para que el LLM narre el combo completo de forma natural.
*   **Filtro Anti-Parpadeo (UI Memory):** Un reloj interno de 4 segundos evita que el OCR genere falsos positivos cuando los textos del juego se quedan brillando en la pantalla.
*   **Sistema de Interrupción Suave (Caster Hype):** Diferencia entre "sucesos menores" (relleno de silencios cada 15 segundos) y "eventos críticos" (*Tetris*, *T-Spin* o peligro alto). Si un evento crítico ocurre mientras la IA habla de algo irrelevante, el sistema corta el audio en la línea de tiempo usando la librería `MoviePy`, aplica un *fadeout* (desvanecimiento) y hace que el comentarista "grite" la nueva jugada al instante, simulando la emoción humana en vivo.

## Tecnologías Utilizadas
*   `Streamlit` (Interfaz Web)
*   `Transformers`, `torch` (Inferencia IA Local)
*   `Huggingface_hub` (Inferencia NLP Cloud)
*   `MoviePy` (Renderizado de Video y Mezcla de Audio a 44100Hz)

---
**Autor:** Jesús Velázquez - Instituto Tecnológico de Culiacán.