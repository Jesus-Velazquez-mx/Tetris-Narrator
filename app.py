# ====== IMPORTS ======
import streamlit as st
import os
import shutil
import tempfile
import time
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip

from captura import ModuloCaptura
from vision import ModuloVision
from narracion import narracion
from audio import ModuloAudio

# ====== CONFIGURACIÓN DE LA INTERFAZ ======
# Define el título de la pestaña en el navegador y el ancho de la página
st.set_page_config(page_title="IA Narrador de Tetris", layout="wide")

# ====== CARGA DE MODELOS DE IA ======
# Se utiliza caché para que los modelos pesados no se recarguen cada vez que el usuario interactúa con la UI
@st.cache_resource
def cargar_modelos():
    vision = ModuloVision()
    narrador = narracion()
    voz = ModuloAudio()
    return vision, narrador, voz

# ====== INTERFAZ PRINCIPAL Y CARGA DE VIDEO ======
st.title("🎮 Narrador IA de Tetris")
st.markdown("Sube un clip de Tetris y observa cómo la IA comenta la jugada generando un video narrado final.")

with st.spinner("Cargando modelos de IA en memoria (esto puede tomar un momento)..."):
    vision, narrador, voz = cargar_modelos()

# Componente para subir el archivo de video
archivo_video = st.file_uploader("Sube tu video de Tetris (.mp4)", type=["mp4"])

# ====== LAYOUT DE COLUMNAS ======
# Si hay un video cargado, divide la pantalla en dos: una para el video y otra para los comentarios en vivo
if archivo_video is not None:
    col_video, col_comentarios = st.columns([1, 1])
    
    with col_video:
        st.subheader("Video Original")
        st.video(archivo_video)
        
    with col_comentarios:
        st.subheader("Comentarios en Vivo")
        caja_comentarios = st.container()

    st.markdown("---")

    # ====== PROCESAMIENTO PRINCIPAL ======
    # Botón que detona toda la lógica de análisis y renderizado
    if st.button("Iniciar Procesamiento, Narración y Renderizado", type="primary"):
        
        # ====== PREPARACIÓN DE ARCHIVOS TEMPORALES ======
        carpeta_trabajo = "frames_temporales_tetris"
        rutas_audios_temporales = []
        clips_de_audio_generados = []
        
        # Guarda el video subido en un archivo temporal para que OpenCV y MoviePy puedan leerlo
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
            tmp_video.write(archivo_video.read())
            ruta_tmp_video = tmp_video.name
        
        try:
            # ====== PASO 1: EXTRACCIÓN DE FRAMES ======
            with st.spinner("Paso 1: Extrayendo frames del video..."):
                intervalo_frames = 1.0
                capturador = ModuloCaptura(ruta_video=ruta_tmp_video, carpeta_salida=carpeta_trabajo, intervalo_segundos=intervalo_frames)
                rutas_frames = capturador.extraer_frames()
            
            if rutas_frames:
                progreso_texto = st.empty()
                barra_progreso = st.progress(0)
                total_frames = len(rutas_frames)
                
                # Variables de estado para la memoria de la IA y control de ritmo del comentarista
                cooldown_hasta = 0.0
                score_anterior = 0  
                eventos_pendientes = []  
                peligro_anterior = "LOW"  
                ultimo_evento_visto = None
                tiempo_ultimo_evento = -999.0
                # Evita que eventos de relleno sean cortados a la mitad
                audio_es_intocable = False
                
                # ====== PASO 2: ANÁLISIS DE VIDEO Y GENERACIÓN DE AUDIO ======
                for idx, ruta_frame in enumerate(rutas_frames):
                    tiempo_actual = idx * intervalo_frames
                    progreso_texto.text(f"Paso 2: Analizando segundo {tiempo_actual} de {total_frames}...")
                    
                    # Llamada al modelo de visión para entender qué pasa en la imagen
                    datos_vision = vision.clasificar_imagen(ruta_frame)
                    
                    if datos_vision:
                        score_actual = datos_vision.get("score", 0)
                        delta_score = score_actual - score_anterior
                        
                        # Evita que la IA repita el mismo evento si ocurre muy rápido (en menos de 4 segundos)
                        evento_frame = datos_vision.get("evento")
                        if evento_frame:
                            if evento_frame == ultimo_evento_visto and (tiempo_actual - tiempo_ultimo_evento) < 4.0:
                                pass 
                            else:
                                if not eventos_pendientes or eventos_pendientes[-1] != evento_frame:
                                    eventos_pendientes.append(evento_frame)
                                # Actualizamos la memoria absoluta para el filtro temporal
                                ultimo_evento_visto = evento_frame
                                tiempo_ultimo_evento = tiempo_actual
                        
                        # Detección de peligro de perder (torre alta) y aumento de puntuación
                        hay_peligro = datos_vision.get("danger") == "HIGH" 
                        hubo_puntos = delta_score >= 100  
                        
                        # Relleno ajustado a 15 segundos (Para que el comentarista no se quede callado mucho tiempo)
                        relleno_tiempo = (idx % 15 == 0) 
                        
                        # Detecta solo el "borde" del peligro (cuando pasa de LOW/MEDIUM a HIGH)
                        peligro_nuevo = hay_peligro and peligro_anterior != "HIGH"
                        
                        # Clasificación de la urgencia de lo que está pasando
                        es_suceso_importante = bool(eventos_pendientes) or peligro_nuevo
                        es_suceso_menor = hubo_puntos or relleno_tiempo
                        
                        eventos_interrumpibles = ["TETRIS", "T-SPIN"] 
                        hay_evento_critico = any(e in eventos_interrumpibles for e in eventos_pendientes)

                        # Usamos peligro_nuevo en lugar de hay_peligro para no interrumpir constantemente
                        debe_interrumpir = hay_evento_critico or peligro_nuevo

                        hubo_interrupcion = False
                        
                        # ====== LÓGICA DE INTERRUPCIÓN DEL COMENTARISTA ======
                        if debe_interrumpir and (tiempo_actual < cooldown_hasta):
                            # Solo interrumpimos si el audio NO es intocable (es decir, es relleno)
                            if not audio_es_intocable:
                                if clips_de_audio_generados:
                                    # Solo asomamos el clip (no le hacemos .pop() todavía)
                                    clip_anterior = clips_de_audio_generados[-1] 
                                    tiempo_hablado = round(tiempo_actual - clip_anterior.start, 2)
                                    
                                    # Obligamos a que la IA lleve hablando al menos 1.5 segundos antes de cortarla
                                    if tiempo_hablado >= 1.5: 
                                        clip_anterior = clips_de_audio_generados.pop() # Ahora sí lo sacamos
                                        fade_dur = min(0.2, tiempo_hablado / 2) # Aplicamos un desvanecimiento suave (fade out)
                                        clip_recortado = clip_anterior.set_duration(tiempo_hablado).audio_fadeout(fade_dur)
                                        clips_de_audio_generados.append(clip_recortado)
                                        
                                        # Liberamos el micrófono
                                        cooldown_hasta = tiempo_actual
                                        hubo_interrupcion = True

                        # ====== GENERACIÓN DE GUION Y VOZ ======
                        # ¿Ocurrió algo Y el micrófono está libre?
                        if (es_suceso_importante or es_suceso_menor) and (tiempo_actual >= cooldown_hasta):
                            
                            evento_final = None
                            if eventos_pendientes:
                                evento_final = " and ".join(eventos_pendientes)
                                
                            # Se pide a la IA generativa (texto) que cree una frase basándose en el contexto del juego
                            comentario = narrador.generar_comentario(
                                descripcion_vision=datos_vision["descripcion"],
                                speed_lv=datos_vision.get("speed_lv", 1),
                                lines=datos_vision.get("lines", 0),
                                score=score_actual,
                                evento=evento_final, 
                                back_to_back=datos_vision.get("back_to_back", False),
                                danger=datos_vision.get("danger", "LOW") 
                            )
                            
                            # Limpieza de eventos una vez comentados
                            eventos_pendientes = []
                            score_anterior = score_actual
                            
                            # Envía el texto a la IA de voz (TTS) para generar el archivo de audio (.wav)
                            nombre_archivo_wav = f"temp_audio_{idx}.wav"
                            ruta_audio = voz.generar_audio(comentario, nombre_archivo_wav)
                            
                            if ruta_audio:
                                tiempo_formateado = time.strftime('%M:%S', time.gmtime(tiempo_actual))
                                
                                # Muestra el comentario de texto y el reproductor de audio en la interfaz
                                with caja_comentarios:
                                    st.markdown(f"**⏱️ {tiempo_formateado}** | 🎙️ _{comentario}_")
                                    st.audio(ruta_audio, format="audio/wav") 
                                    rutas_audios_temporales.append(ruta_audio)
                                
                                # Ajusta el tiempo de inicio del nuevo audio (agregando retraso si hubo interrupción)
                                tiempo_arranque = tiempo_actual + 0.3 if hubo_interrupcion else tiempo_actual
                                
                                clip = AudioFileClip(ruta_audio).set_start(tiempo_arranque)
                                clips_de_audio_generados.append(clip)
                                
                                cooldown_hasta = tiempo_arranque + clip.duration

                                audio_es_intocable = es_suceso_importante
                                
                        # Actualizamos el estado del peligro para el siguiente frame (CORRECCIÓN 3)
                        peligro_anterior = datos_vision.get("danger", "LOW")
                                    
                    # Actualiza la barra de progreso en la UI
                    barra_progreso.progress((idx + 1) / total_frames)
                
                progreso_texto.text("¡Análisis finalizado! Iniciando renderizado de video...")
                
                # ====== PASO 3: RENDERIZADO FINAL DEL VIDEO ======
                if clips_de_audio_generados:
                    with st.spinner("🎬 Paso 3: Montando la transmisión final (Audio exclusivo IA)..."):
                        video_original = VideoFileClip(ruta_tmp_video)
                        
                        # Baja el volumen del juego original al 20% para que resalte la voz del comentarista
                        base_silenciosa = video_original.audio.volumex(0.2) if video_original.audio else None
                        
                        # Mezcla los audios
                        if base_silenciosa:
                            audio_final = CompositeAudioClip([base_silenciosa] + clips_de_audio_generados)
                        else:
                            audio_final = CompositeAudioClip(clips_de_audio_generados)
                            
                        # Acopla el audio mezclado al video original
                        video_narrado = video_original.set_audio(audio_final)
                        ruta_video_salida = "video_final_esports.mp4"
                        
                        # Exporta el resultado final con ajustes optimizados para renderizado rápido (ultrafast)
                        video_narrado.write_videofile(
                            ruta_video_salida, 
                            codec="libx264", 
                            audio_codec="aac", 
                            fps=24, 
                            audio_fps=44100, 
                            preset="ultrafast", 
                            logger=None
                        )                        
                        video_original.close()
                        video_narrado.close()
                        
                        # Muestra el resultado final en la interfaz
                        st.success("✨ ¡Transmisión renderizada con éxito!")
                        st.video(ruta_video_salida)

        except Exception as e:
            st.error(f"Ocurrió un error en la ejecución: {e}")

        # ====== LIMPIEZA DE ARCHIVOS TEMPORALES ======
        # El bloque 'finally' asegura que, pase lo que pase (incluso si hay errores), los archivos basura se borren
        finally:
            with st.spinner("Limpiando archivos temporales..."):
                # Forzamos el cierre en memoria de todos los clips generados antes de borrarlos para evitar errores de archivo en uso
                for clip in clips_de_audio_generados:
                    try:
                        clip.close()
                    except:
                        pass
                
                if os.path.exists(carpeta_trabajo): shutil.rmtree(carpeta_trabajo)
                if os.path.exists(ruta_tmp_video): os.remove(ruta_tmp_video)
                for ruta_wav in rutas_audios_temporales:
                    if os.path.exists(ruta_wav):
                        try:
                            os.remove(ruta_wav)
                        except:
                            pass 
            st.info("🧹 Sistema limpio: Archivos temporales eliminados.")