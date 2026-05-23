import streamlit as st
import os
import shutil
import tempfile
import time

# Importamos las clases de tus módulos
from captura import ModuloCaptura
from vision import ModuloVision
from narracion import narracion

# 1. Configuración de la página
st.set_page_config(page_title="IA Narrador de Tetris", layout="wide")

# 2. Cargar modelos en caché para que no se reinicien en cada clic
@st.cache_resource
def cargar_modelos():
    vision = ModuloVision()
    narrador = narracion()
    return vision, narrador

st.title("🎮 Narrador IA de Tetris")
st.markdown("Sube un clip de Tetris y observa cómo la IA comenta la jugada en base a Visión y OCR.")

# Inicializamos los modelos mostrando un mensaje de carga en la pantalla
with st.spinner("Cargando modelos de IA en memoria (esto puede tomar un momento)..."):
    vision, narrador = cargar_modelos()

# 3. Interfaz de subida de video
archivo_video = st.file_uploader("Sube tu video de Tetris (.mp4)", type=["mp4"])

if archivo_video is not None:
    # Dividimos la pantalla en dos columnas
    col_video, col_comentarios = st.columns([1, 1])
    
    with col_video:
        st.subheader("Video Original")
        # Reproductor nativo de Streamlit
        st.video(archivo_video)
        
    with col_comentarios:
        st.subheader("Comentarios en Vivo")
        # Contenedor vacío donde iremos imprimiendo los comentarios
        caja_comentarios = st.container()

        # ==========================================
    # 🛠️ PANEL DE DEBUG
    # ==========================================
    st.markdown("---")
    st.subheader("🛠️ Panel de Debug")
    st.caption("Prueba cada módulo de forma individual.")
    
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        btn_debug_captura = st.button("📸 Probar Captura", use_container_width=True)
    with col_btn2:
        btn_debug_vision = st.button("👁️ Probar Visión", use_container_width=True)
    with col_btn3:
        btn_debug_narracion = st.button("🗣️ Probar Narración", use_container_width=True)

    # ------------------------------------------
    # LÓGICA DE DEBUG: MÓDULO DE CAPTURA
    # ------------------------------------------
    if btn_debug_captura:
        carpeta_debug = "frames_debug"
        
        # Guardamos el video temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
            tmp_video.write(archivo_video.read())
            ruta_tmp_video = tmp_video.name
            
        try:
            with st.spinner("Ejecutando ModuloCaptura..."):
                capturador_debug = ModuloCaptura(ruta_video=ruta_tmp_video, carpeta_salida=carpeta_debug, intervalo_segundos=1.0)
                rutas_frames_debug = capturador_debug.extraer_frames()
                
            if rutas_frames_debug:
                st.success(f"¡Captura exitosa! Se extrajeron {len(rutas_frames_debug)} frames en total.")
                
                # Mostramos los primeros 3 frames como prueba visual
                st.write("**Muestra de los primeros 3 frames:**")
                cols_muestras = st.columns(min(3, len(rutas_frames_debug)))
                for i, col_img in enumerate(cols_muestras):
                    col_img.image(rutas_frames_debug[i], caption=f"Frame {i+1}", use_container_width=True)
            else:
                st.error("El módulo se ejecutó, pero no devolvió ningún frame.")
                
        except Exception as e:
            st.error(f"Error en el módulo de captura: {e}")
            
        finally:
            # Limpieza del debug
            if os.path.exists(carpeta_debug):
                shutil.rmtree(carpeta_debug)
            if os.path.exists(ruta_tmp_video):
                os.remove(ruta_tmp_video)

    # ------------------------------------------
    # (Aquí iría después la lógica de Visión y Narración)
    # ------------------------------------------
# ------------------------------------------
    # LÓGICA DE DEBUG: MÓDULO DE VISIÓN
    # ------------------------------------------
    if btn_debug_vision:
        if archivo_video is None:
            st.warning("⚠️ Por favor, sube un video primero para extraer un frame de prueba.")
        else:
            carpeta_debug_vision = "frame_debug_vision"
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
                tmp_video.write(archivo_video.read())
                ruta_tmp_video = tmp_video.name
                
            try:
                with st.spinner("Extrayendo 1 frame y ejecutando BLIP/OCR (esto tomará unos segundos)..."):
                    import cv2
                    if not os.path.exists(carpeta_debug_vision):
                        os.makedirs(carpeta_debug_vision)
                    
                    ruta_frame_prueba = os.path.join(carpeta_debug_vision, "frame_prueba.jpg")
                    
                    # Extraemos rápido solo el primer frame usando OpenCV
                    cap = cv2.VideoCapture(ruta_tmp_video)
                    ret, frame = cap.read()
                    cap.release()
                    
                    if ret:
                        cv2.imwrite(ruta_frame_prueba, frame)
                        
                        # Mostramos el frame en la interfaz web
                        st.markdown("**Frame analizado:**")
                        st.image(ruta_frame_prueba, width=500)
                        
                        # Mandamos llamar a tu módulo
                        resultados = vision.clasificar_imagen(ruta_frame_prueba)
                        
                        if resultados:
                            st.success("¡Análisis visual completado con éxito!")
                            # st.json crea una tarjeta muy visual para leer diccionarios en Streamlit
                            st.json(resultados) 
                        else:
                            st.error("El modelo no devolvió datos.")
                    else:
                        st.error("No se pudo extraer el frame del video.")
            except Exception as e:
                st.error(f"Error en el módulo de visión: {e}")
            finally:
                # Limpiamos todo al terminar
                if os.path.exists(carpeta_debug_vision):
                    shutil.rmtree(carpeta_debug_vision)
                if os.path.exists(ruta_tmp_video):
                    os.remove(ruta_tmp_video)
        
  # ------------------------------------------
    # LÓGICA DE DEBUG: MÓDULO DE NARRACIÓN
    # ------------------------------------------
    if btn_debug_narracion:
        st.write("Generando un comentario de prueba con datos simulados...")
        try:
            with st.spinner("Creando narrativa..."):
                # Simulamos un escenario de alta tensión para ver cómo reacciona el narrador
                comentario_prueba = narrador.generar_comentario(
                    descripcion_vision="A Tetris board with a very high stack, close to top out.",
                    speed_lv=15,
                    lines=40,
                    score=250000,
                    evento="TETRIS",
                    danger="HIGH",
                    back_to_back=True
                )
                
                if comentario_prueba:
                    st.success("¡Módulo de narración ejecutado correctamente!")
                    st.info(f"**Escenario simulado:** Nivel 15 | Peligro: ALTO | Evento: TETRIS (Back-to-Back)")
                    st.markdown(f"🎙️ **Narrador dice:** _{comentario_prueba}_")
                else:
                    st.error("El módulo no devolvió ningún comentario.")
        except Exception as e:
            st.error(f"Error en el módulo de narración: {e}")

    st.markdown("---")

    if st.button("Iniciar Procesamiento y Narración", type="primary"):
        # Configuramos la carpeta que se creará y luego se borrará
        carpeta_trabajo = "frames_temporales_tetris"
        
        # Streamlit guarda los archivos en memoria. Necesitamos un archivo físico temporal para OpenCV.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
            tmp_video.write(archivo_video.read())
            ruta_tmp_video = tmp_video.name
        
        try:
            with st.spinner("Extrayendo frames del video..."):
                capturador = ModuloCaptura(ruta_video=ruta_tmp_video, carpeta_salida=carpeta_trabajo, intervalo_segundos=1.0)
                rutas_frames = capturador.extraer_frames()
            
            if not rutas_frames:
                st.error("No se pudieron extraer frames. Revisa el video.")
            else:
                progreso_texto = st.empty()
                barra_progreso = st.progress(0)
                total_frames = len(rutas_frames)
                transcripcion_final = []
                
                # Iteramos sobre cada imagen extraída
                for idx, ruta_frame in enumerate(rutas_frames):
                    progreso_texto.text(f"Analizando frame {idx + 1} de {total_frames}...")
                    
                    # A. Pasamos la imagen por el módulo de visión
                    datos_vision = vision.clasificar_imagen(ruta_frame)
                    
                    if datos_vision:
                        # --- NUEVA LÓGICA DE HEURÍSTICA ---
                        # Solo narramos si ocurre un evento, si hay peligro, o cada 6 segundos para rellenar
                        hay_evento = datos_vision.get("evento") is not None
                        hay_peligro = datos_vision.get("danger") == "HIGH" 
                        relleno_tiempo = (idx % 6 == 0) 
                        
                        if hay_evento or hay_peligro or relleno_tiempo:
                            # B. Pasamos los datos al LLM (ahora sí generará textos largos)
                            comentario = narrador.generar_comentario(
                                descripcion_vision=datos_vision["descripcion"],
                                speed_lv=datos_vision.get("speed_lv", 1),
                                lines=datos_vision.get("lines", 0),
                                score=datos_vision.get("score", 0),
                                evento=datos_vision.get("evento", None),
                                back_to_back=datos_vision.get("back_to_back", False),
                                danger="LOW" 
                            )
                            
                            # C. Mostramos el resultado
                            with caja_comentarios:
                                st.markdown(f"**⏱️ 00:{idx:02d}** | 🎙️ _{comentario}_")
                                transcripcion_final.append(f"00:{idx:02d} | {comentario}")
                        else:
                            # Si no pasa nada interesante, lo ignoramos y no saturamos la API
                            print(f"Frame 00:{idx:02d} - Sin eventos, saltando narración...")
                    
                    # Actualizamos la barra de carga
                    barra_progreso.progress((idx + 1) / total_frames)
                
                progreso_texto.text("¡Procesamiento finalizado!")
                st.success("Narración completada con éxito.")

        except Exception as e:
            st.error(f"Ocurrió un error en la ejecución: {e}")

        finally:
            # 4. Limpieza absoluta
            with st.spinner("Limpiando archivos temporales..."):
                if os.path.exists(carpeta_trabajo):
                    shutil.rmtree(carpeta_trabajo)
                if os.path.exists(ruta_tmp_video):
                    os.remove(ruta_tmp_video)
            st.info("🧹 Sistema limpio: Se ha borrado la carpeta de imágenes y el video temporal.")