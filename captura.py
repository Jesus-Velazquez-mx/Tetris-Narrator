# ====== IMPORTS ======
import cv2
import os

class ModuloCaptura:

    # ====== INICIALIZAR ESTADO Y RUTAS ======
    # Configura las rutas de video, carpetas de salida y crea directorios si no existen
    def __init__(self, ruta_video, carpeta_salida="frames_capturados", intervalo_segundos=1):
        self.ruta_video = ruta_video
        self.carpeta_salida = carpeta_salida
        self.intervalo_segundos = intervalo_segundos
        
        # Crear la carpeta de salida automáticamente al instanciar la clase
        if not os.path.exists(self.carpeta_salida):
            os.makedirs(self.carpeta_salida)
            print(f"Carpeta de salida lista: {self.carpeta_salida}")

    # ====== EXTRACCIÓN DE FRAMES ======
    # Procesa el video de entrada y extrae imágenes iterativamente según el intervalo
    def extraer_frames(self):
        
        #Retorna una lista con las rutas de los archivos guardados.
        cap = cv2.VideoCapture(self.ruta_video)
        
        if not cap.isOpened():
            print(f"Error: No se pudo abrir el video en {self.ruta_video}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        frames_a_saltar = int(fps * self.intervalo_segundos)
        
        contador_total = 0
        frames_guardados = 0
        rutas_frames = [] # Lista para guardar dónde quedó cada imagen

        print("Iniciando extracción de frames...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break 
            
            if contador_total % frames_a_saltar == 0:
                nombre_archivo = os.path.join(self.carpeta_salida, f"frame_{frames_guardados:04d}.jpg")
                cv2.imwrite(nombre_archivo, frame)
                rutas_frames.append(nombre_archivo) # Guardamos la ruta
                frames_guardados += 1
            
            contador_total += 1

        cap.release()
        print(f"Proceso completado. {frames_guardados} frames listos para analizar.")
        
        return rutas_frames

# =========================================================
# PRUEBA DEL MÓDULO (MAIN)
# =========================================================
if __name__ == "__main__":
    # Instanciamos la clase
    capturador = ModuloCaptura(ruta_video="C:\\Users\\jesu1\\Videos\\test.mp4", intervalo_segundos=0.5)
    
    # Ejecutamos el método
    lista_de_imagenes = capturador.extraer_frames()
    
    # Imprimimos las primeras 3 rutas para verificar
    print("Primeras imágenes guardadas:", lista_de_imagenes[:3])